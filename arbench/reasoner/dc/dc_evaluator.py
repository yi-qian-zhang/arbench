import json
import os
import random
import re
import sys
import string
from collections import deque
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

from arbench.reasoner.dc.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_dc import (
    ANSWER_CHOICES,
    CHOICE_TO_INDEX,
    convert_initial_info_to_string,
    extract_answer_choice,
    format_choices,
    calculate_accuracy,
    is_valid_choice,
    choice_to_index,
    index_to_choice,
    extract_reasoning,
    CrimeDetectionSearchTreeNode
)
from fire import Fire
from dotenv import load_dotenv

# new method
from arbench.utils.interactive_policy_client import InteractivePolicyClient

load_dotenv()

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")

ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
CHOICE_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "": ""}

METHOD_DICT = {
        "zero_shot": propose_template,
        "few_shot": propose_template_with_1_shot,
        "few_shot_inst": propose_template_with_1_shot_inst,
    }


def extract_answer_choice(input_string: str) -> str:
    match = re.search(r"Answer:\s*(.*)", input_string)
    if not match:
        return ""
    
    try:
        answer = match.group(1)
        choices = re.findall(r"A|B|C|D|E", answer)
        return choices[-1] if choices else ""
    except:
        print(f"==> input_string: {input_string}, ==> answer: {answer}. ==> choices: {choices}")
        raise ValueError("Failed to extract answer choice")


# Custom parse_keypoints for CD game (different from utils version)
def parse_keypoints(input_string: str) -> List[int]:
    input_string = input_string.lower()
    matches = re.findall(r"hit point: ([\d, ]+)", input_string)
    matches = [item for item in matches if item.strip()]
    
    numbers = []
    try:
        if matches:
            numbers = [int(num.strip()) for num in matches[0].split(",")]
    except:
        print(input_string, matches)
        raise ValueError("Failed to parse keypoints")
        
    return numbers


# Aliases for consistency
format_keypoints = lambda keypoints: "\n".join(f"{i+1}. {key}" for i, key in enumerate(keypoints)) + "\n"


def remove_commas(string: str) -> str:
    return string.replace(",", "") if "," in string else string


def remove_punctuation_at_ends(input_str: str) -> str:
    punctuations = string.punctuation
    if input_str and input_str[0] in punctuations:
        input_str = input_str[1:]
    if input_str and input_str[-1] in punctuations:
        input_str = input_str[:-1]
    return input_str


def convert_tree_to_json(node: 'SearchTreeNode') -> List[Dict]:
    if not node:
        return []

    result = []
    queue = deque([(node, 0)])

    while queue:
        current_node, level = queue.popleft()

        result.append({
            "name": current_node.suspect,
            "question": current_node.question,
            "feedback": current_node.answer,
            "level": level,
            "value": current_node.value,
            "record": current_node.step_record,
            "chosen": current_node.chosen,
        })

        for child in current_node.children:
            queue.append((child, level + 1))

    return result


class SearchTreeNode:
    
    def __init__(self, parent: Optional['SearchTreeNode'] = None):
        # Basic attributes
        self.suspect = ""  # Name of suspect being questioned
        self.question = ""  # Question asked to suspect
        self.answer = ""  # Answer received from suspect
        self.value = -1  # Node evaluation score
        self.children: List['SearchTreeNode'] = []
        self.parent = parent
        self.depth = 0
        self.total_value = 0  # Not used in greedy search
        self.visits = 0  # Not used in greedy search
        self.chosen = False

        # Output attributes for logging
        self.input_token = 0
        self.output_token = 0
        self.question_hit_cnt = 0
        self.right_question_list: List[str] = []
        self.matched_keyquestion_set: set = set()

        # Attributes for step sampling
        self.choice_suspect_prompt = ""
        self.ask_prompt = ""
        self.step_record: List[Dict] = []

    def add_child(self, child_node: 'SearchTreeNode') -> None:
        child_node.depth = self.depth + 1
        child_node.question_hit_cnt = self.question_hit_cnt
        child_node.right_question_list = self.right_question_list
        child_node.matched_keyquestion_set = self.matched_keyquestion_set
        self.children.append(child_node)

    def get_conversation_record(self) -> List[Dict]:
        record = []
        current_node = self
        
        # Traverse back to root, collecting conversation records
        while current_node.parent and current_node.parent.depth > 0:
            parent_info = {
                "suspect": current_node.parent.suspect,
                "question": current_node.parent.question,
                "feedback": current_node.parent.answer,
            }
            record.insert(0, parent_info)
            current_node = current_node.parent
            
        return record

    def display_node_info(self) -> None:
        print("Suspect:", self.suspect)
        print("Question:", self.question)
        print("Answer:", self.answer)
        print("Value:", self.value)
        print("Input Token:", self.input_token)
        print("Output Token:", self.output_token)
        print("Visits:", self.visits)
        print("Depth:", self.depth)
        print("Total Value:", self.total_value)
        print("Question Hit Count:", self.question_hit_cnt)
        print("Right Question List:", self.right_question_list)
        print("Matched Keyquestion Set:", self.matched_keyquestion_set)

    def calculate_token_sums(self) -> Tuple[int, int]:
        input_token_sum = 0
        output_token_sum = 0
        current_node = self

        while current_node:
            input_token_sum += current_node.input_token
            output_token_sum += current_node.output_token
            current_node = current_node.parent

        return input_token_sum, output_token_sum


def expand_node(
    node: SearchTreeNode,
    branch: int,
    current_round: int,
    max_round: int,
    init_info: str,
    suspect_name_str: str,
    suspects: List[Dict],
    key_question_dict: Dict[str, List[str]],
    model: str,
    response_model: str,
    policy_temperature: float,
    policy_top_p: float,
    response_temperature: float,
    response_top_p: float
) -> None:
    # Initialize evaluation agents for suspects with key questions
    keypoint_eval_agents = {
        item["name"]: [{
            "role": "system",
            "content": keypoint_hits_prompt.format(
                question=init_info,
                name=item["name"],
                answer=item["story"],
                keypoints=format_keypoints(item["key_question"]),
            ),
        }]
        for item in suspects
        if "key_question" in item
    }
    
    # Initialize response agents for role-playing
    response_agents = {
        item["name"]: [{
            "role": "system",
            "content": respond_template.format(
                name=item["name"], task=item["task"], story=item["story"]
            ),
        }]
        for item in suspects
    }

    idx = 0
    while idx < branch:
        # Select suspect
        choice_suspect_prompt = select_suspect_template.format(
            turn=current_round, suspect_names=suspect_name_str
        )
        choice_suspect_agent = [{"role": "user", "content": choice_suspect_prompt}]
        
        try:
            response = inference(choice_suspect_agent, model=model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            selected_suspect = response.choices[0].message.content
            selected_suspect = remove_punctuation_at_ends(selected_suspect)

            assert selected_suspect in response_agents.keys(), \
                f"{selected_suspect} is not in {response_agents.keys()}"
                
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            choice_suspect_agent.append({"role": "assistant", "content": selected_suspect})
            choice_suspect_agent.append({"role": "user", "content": refine_select_suspect_prompt})
            continue

        # Only increment when valid suspect is selected
        idx += 1

        # Create new child node
        current_node = SearchTreeNode(node)
        node.add_child(current_node)

        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.suspect = selected_suspect
        
        # Get conversation record
        record = current_node.get_conversation_record()
        current_node.record = record

        # Format record for prompt
        record_str = ""
        for entity in record:
            record_str += (
                f"Question for {entity['suspect']}: {entity['question']} "
                f"Feedback: {entity['feedback']}\n"
            )

        # Generate question for suspect
        ask_agent = [{
            "role": "system",
            "content": propose_template.format(turn=max_round, background=init_info),
        }]
        
        ask_agent.append({
            "role": "user",
            "content": question_propose_prompt_searching.format(
                record=record_str, suspect=selected_suspect
            ),
        })
        
        response = inference(ask_agent, model=model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        ask_prompt = (
            f"system: {propose_template.format(turn=max_round, background=init_info)}\n"
            f"user: {question_propose_prompt.format(turn=current_round, record=record, suspect=selected_suspect)}"
        )
        question = response.choices[0].message.content

        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.question = question

        # Evaluate question quality if suspect has key questions
        if selected_suspect in keypoint_eval_agents:
            keypoint_eval_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(keypoint_eval_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            check_results = response.choices[0].message.content
            current_node.input_token += response.usage.prompt_tokens
            current_node.output_token += response.usage.completion_tokens

            numbers = parse_keypoints(check_results)
            current_node.value = len(numbers)

            if numbers:
                try:
                    current_node.question_hit_cnt += 1
                    current_node.right_question_list.append(question)
                    for num in numbers:
                        current_node.matched_keyquestion_set.add(
                            key_question_dict[selected_suspect][num - 1]
                        )
                except IndexError:
                    pass

        current_node.ask_prompt = ask_prompt
        current_node.choice_suspect_prompt = choice_suspect_prompt
        current_node.step_record = record

        # Get suspect response
        response_agents[selected_suspect].append({"role": "user", "content": question})
        response = inference(response_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        suspect_response = response.choices[0].message.content
        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.answer = suspect_response

        response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})


def select_best_child_node(node: SearchTreeNode) -> Optional[SearchTreeNode]:
    if not node.children:
        return None

    max_value = max(child.value for child in node.children)
    max_value_children = [child for child in node.children if child.value == max_value]
    best_node = random.choice(max_value_children)
    best_node.chosen = True
    
    return best_node


def _run_greedy_evaluation(
    dataset: List[Dict],
    logs: List[Dict], 
    output_path: str,
    policy_model: str,
    max_turn: int,
    branch: int,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float
) -> None:
    """Run evaluation using greedy search method."""
    tree_logs = []
    tree_path = output_path.replace(".json", "") + "_tree.json"
    
    if os.path.exists(tree_path):
        with open(tree_path, "r", encoding="utf-8") as file:
            tree_logs = json.load(file)

    for i in tqdm(range(len(logs), len(dataset))):
        # Prepare case information
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]
        
        choice_str = ", ".join([
            f"{index}. {item['name']}"
            for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
        ])
        
        key_question_dict = {
            item["name"]: item["key_question"]
            for item in dataset[i]["suspects"]
            if "key_question" in item
        }
        
        suspect_name_str = ", ".join([
            item["name"] for item in dataset[i]["initial_information"]["suspect"]
        ])

        # Initialize search tree and perform greedy search
        root = SearchTreeNode(None)
        current_node = root
        
        for j in range(max_turn):
            expand_node(
                node=current_node,
                branch=branch,
                current_round=j + 1,
                max_round=max_turn,
                init_info=init_info,
                suspect_name_str=suspect_name_str,
                suspects=dataset[i]["suspects"],
                key_question_dict=key_question_dict,
                model=policy_model,
                response_model=response_model,
                policy_temperature=policy_temperature,
                policy_top_p=policy_top_p,
                response_temperature=response_temperature,
                response_top_p=response_top_p
            )

            new_child_node = select_best_child_node(current_node)
            current_node = new_child_node

        # Get final prediction
        final_node = current_node
        record = final_node.get_conversation_record()
        record.append({
            "suspect": final_node.suspect,
            "question": final_node.question,
            "feedback": final_node.answer,
        })
        
        record_str = ""
        for entity in record:
            record_str += (
                f"Question for {entity['suspect']}: {entity['question']} "
                f"Feedback: {entity['feedback']}\n"
            )

        select_murderer_prompt = (
            propose_template.format(turn=max_turn, background=init_info) +
            select_murderer_template_searching.format(record=record_str, choice=choice_str)
        )
        
        select_murderer_agent = [{"role": "user", "content": select_murderer_prompt}]
        response = inference(select_murderer_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content

        get_result_input_token = response.usage.prompt_tokens
        get_result_output_token = response.usage.completion_tokens

        pred = CHOICE_TO_INDEX[extract_answer_choice(raw_pred).strip()]
        
        input_token, output_token = final_node.calculate_token_sums()
        
        logs.append({
            "idx": i,
            "raw_pred": raw_pred,
            "pred": pred,
            "label": label,
            "round": max_turn,
            "input_token_sum": input_token + get_result_input_token,
            "output_token_sum": output_token + get_result_output_token,
            "correctness": pred == label,
            "record": record,
        })
        
        # Save logs
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=4)

        # Save tree
        tree = convert_tree_to_json(root)
        tree_logs.append(tree)
        with open(tree_path, "w", encoding="utf-8") as file:
            json.dump(tree_logs, file, indent=4)

def _run_traditional_evaluation_interactive(
    method: str,
    dataset: List[Dict],
    logs: List[Dict],
    output_path: str,
    policy_client: InteractivePolicyClient,
    response_model: str,
    response_temperature: float,
    response_top_p: float,
    max_turn: int,
) -> None:
    """使用交互式策略运行传统评估方法 (修改版：增加full_completions日志)"""

    for i in tqdm(range(len(logs), len(dataset))):
        # 准备案例信息
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]

        total_input_token, total_output_token = 0, 0
        choice_str = ", ".join([
            f"{index}. {item['name']}"
            for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
        ])

        # 系统提示
        system_content = METHOD_DICT[method].format(background=init_info, turn=max_turn)

        # 初始化嫌疑人的响应代理
        response_agents = {
            item["name"]: [{
                "role": "system",
                "content": respond_template.format(
                    name=item["name"], task=item["task"], story=item["story"]
                ),
            }]
            for item in dataset[i]["suspects"]
        }
        
        suspect_name_str = ", ".join([
            item["name"] for item in dataset[i]["initial_information"]["suspect"]
        ])

        # 存储完整对话记录
        propose_agent = [{
            "role": "system",
            "content": system_content,
        }]
        
        # --- 修改开始: 初始化日志列表 ---
        all_interactions = []
        all_completions = []  # <-- 1. 新增这个列表
        # --- 修改结束 ---

        # 问答循环
        for turn in range(max_turn):
            # 选择嫌疑人（使用交互）
            select_prompt = select_suspect_template.format(
                turn=turn + 1, suspect_names=suspect_name_str
            )
            
            propose_agent.append({
                "role": "user",
                "content": select_prompt,
            })
            
            # 获取之前的对话记录 (用于给响应模型提供上下文)
            record_str = ""
            for j, msg in enumerate(propose_agent):
                if msg["role"] == "assistant" and j > 0:
                    record_str += f"{msg['content']}\n"
            
            # 交互式选择嫌疑人
            selected_suspect, suspect_completion, suspect_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
                system_prompt=system_content,
                user_prompt=select_prompt,
                response_model=response_model,
                response_api_key=RESPONSE_API_KEY,
                response_base_url=RESPONSE_BASE_URL,
                response_temperature=response_temperature,
                response_top_p=response_top_p,
                case_context={
                    'case_info': init_info,
                    'suspects': suspect_name_str,
                    'turn': turn + 1,
                    'record': record_str
                }
            )
            
            total_input_token += input_tok
            total_output_token += output_tok
            all_interactions.extend(suspect_interactions)
            
            # --- 修改开始: 记录完整的completion ---
            all_completions.append({
                "step": f"turn_{turn + 1}_select_suspect",
                "completion": suspect_completion
            })
            # --- 修改结束 ---
            
            selected_suspect = remove_punctuation_at_ends(selected_suspect)
            
            # 验证嫌疑人
            if selected_suspect not in response_agents.keys():
                print(f"Invalid suspect: {selected_suspect}, retrying...")
                propose_agent.append({"role": "assistant", "content": selected_suspect}) # 记录错误答案
                propose_agent.append({"role": "user", "content": refine_select_suspect_prompt})
                continue
            
            # (保持不变: 记录干净答案，以维持逻辑上下文)
            propose_agent.append({"role": "assistant", "content": selected_suspect})
            
            # 生成问题（使用交互）
            question_prompt = question_propose_prompt.format(turn=turn + 1)
            propose_agent.append({
                "role": "user",
                "content": question_prompt,
            })
            
            question, question_completion, question_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
                system_prompt=system_content,
                user_prompt=question_prompt,
                response_model=response_model,
                response_api_key=RESPONSE_API_KEY,
                response_base_url=RESPONSE_BASE_URL,
                response_temperature=response_temperature,
                response_top_p=response_top_p,
                case_context={
                    'case_info': init_info,
                    'suspect': selected_suspect,
                    'turn': turn + 1,
                    'record': record_str  # 传递之前的干净历史
                }
            )
            
            total_input_token += input_tok
            total_output_token += output_tok
            all_interactions.extend(question_interactions)
            
            # --- 修改开始: 记录完整的completion ---
            all_completions.append({
                "step": f"turn_{turn + 1}_generate_question",
                "completion": question_completion
            })
            # --- 修改结束 ---
            
            # (保持不变: 记录干净问题，用于和模拟器交互)
            propose_agent.append({"role": "assistant", "content": question})

            # 获取嫌疑人回复（使用原始inference）
            response_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(
                response_agents[selected_suspect], 
                model=response_model, 
                temperature=response_temperature, 
                top_p=response_top_p, 
                api_key=RESPONSE_API_KEY, 
                base_url=RESPONSE_BASE_URL
            )
            suspect_response = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})
            # (保持不变: 记录干净的用户回复)
            propose_agent.append({"role": "user", "content": suspect_response})

        # 获取最终预测（使用交互）
        final_prompt = select_murderer_template.format(choice=choice_str)
        propose_agent.append({
            "role": "user",
            "content": final_prompt,
        })
        
        raw_pred, pred_completion, pred_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
            system_prompt=system_content,
            user_prompt=final_prompt,
            response_model=response_model,
            response_api_key=RESPONSE_API_KEY,
            response_base_url=RESPONSE_BASE_URL,
            response_temperature=response_temperature,
            response_top_p=response_top_p,
            case_context={
                'case_info': init_info,
                'choices': choice_str,
                'record': record_str # 传递最后一次的历史记录
            }
        )
        
        total_input_token += input_tok
        total_output_token += output_tok
        all_interactions.extend(pred_interactions)
        
        # --- 修改开始: 记录完整的completion ---
        all_completions.append({
            "step": "final_prediction",
            "completion": pred_completion
        })
        # --- 修改结束 ---
        
        # (保持不变: 记录最终裁减的答案)
        propose_agent.append({"role": "assistant", "content": raw_pred})

        pred = CHOICE_TO_INDEX.get(extract_answer_choice(raw_pred), "")
        if pred == "":
            pred = CHOICE_TO_INDEX.get(extract_answer_choice(raw_pred.strip()), "") # 再次尝试

        # --- 修改开始: 在最终日志中添加新键 ---
        logs.append({
            "idx": i,
            "record": propose_agent,         # 干净的高层逻辑日志
            "interactions": all_interactions, # 内部的 <asking>/<response> 日志
            "full_completions": all_completions, # <--- 3. 你想要的带<think>的完整日志
            "respond_conversation": [
                {"name": key, "conversation": value}
                for key, value in response_agents.items()
            ],
            "pred": pred,
            "label": label,
            "round": max_turn,
            "input_token_sum": total_input_token,
            "output_token_sum": total_output_token,
            "correctness": pred == label,
        })
        # --- 修改结束 ---
        
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=4, ensure_ascii=False) # 添加 ensure_ascii=False 以防中文乱码
        
        print(f"Case {i}: Pred={pred}, Label={label}, Correct={pred==label}")

# def _run_traditional_evaluation_interactive(
#     method: str,
#     dataset: List[Dict],
#     logs: List[Dict],
#     output_path: str,
#     policy_client: InteractivePolicyClient,
#     response_model: str,
#     response_temperature: float,
#     response_top_p: float,
#     max_turn: int,
# ) -> None:
#     """使用交互式策略运行传统评估方法"""

#     for i in tqdm(range(len(logs), len(dataset))):
#         # 准备案例信息
#         init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
#         label = dataset[i]["label"]

#         total_input_token, total_output_token = 0, 0
#         choice_str = ", ".join([
#             f"{index}. {item['name']}"
#             for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
#         ])

#         # 系统提示
#         system_content = METHOD_DICT[method].format(background=init_info, turn=max_turn)

#         # 初始化嫌疑人的响应代理
#         response_agents = {
#             item["name"]: [{
#                 "role": "system",
#                 "content": respond_template.format(
#                     name=item["name"], task=item["task"], story=item["story"]
#                 ),
#             }]
#             for item in dataset[i]["suspects"]
#         }
        
#         suspect_name_str = ", ".join([
#             item["name"] for item in dataset[i]["initial_information"]["suspect"]
#         ])

#         # 存储完整对话记录
#         propose_agent = [{
#             "role": "system",
#             "content": system_content,
#         }]
#         all_interactions = []

#         # 问答循环
#         for turn in range(max_turn):
#             # 选择嫌疑人（使用交互）
#             select_prompt = select_suspect_template.format(
#                 turn=turn + 1, suspect_names=suspect_name_str
#             )
            
#             propose_agent.append({
#                 "role": "user",
#                 "content": select_prompt,
#             })
            
#             # 获取之前的对话记录
#             record_str = ""
#             for j, msg in enumerate(propose_agent):
#                 if msg["role"] == "assistant" and j > 0:
#                     record_str += f"{msg['content']}\n"
            
#             # 交互式选择嫌疑人
#             selected_suspect, suspect_completion, suspect_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
#                 system_prompt=system_content,
#                 user_prompt=select_prompt,
#                 response_model=response_model,
#                 response_api_key=RESPONSE_API_KEY,
#                 response_base_url=RESPONSE_BASE_URL,
#                 response_temperature=response_temperature,
#                 response_top_p=response_top_p,
#                 case_context={
#                     'case_info': init_info,
#                     'suspects': suspect_name_str,
#                     'turn': turn + 1,
#                     'record': record_str
#                 }
#             )
            
#             total_input_token += input_tok
#             total_output_token += output_tok
#             all_interactions.extend(suspect_interactions)
            
#             selected_suspect = remove_punctuation_at_ends(selected_suspect)
            
#             # 验证嫌疑人
#             if selected_suspect not in response_agents.keys():
#                 print(f"Invalid suspect: {selected_suspect}, retrying...")
#                 propose_agent.append({"role": "assistant", "content": selected_suspect})
#                 propose_agent.append({"role": "user", "content": refine_select_suspect_prompt})
#                 continue
            
#             propose_agent.append({"role": "assistant", "content": selected_suspect})
            
#             # 生成问题（使用交互）
#             question_prompt = question_propose_prompt.format(turn=turn + 1)
#             propose_agent.append({
#                 "role": "user",
#                 "content": question_prompt,
#             })
            
#             question, question_completion, question_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
#                 system_prompt=system_content,
#                 user_prompt=question_prompt,
#                 response_model=response_model,
#                 response_api_key=RESPONSE_API_KEY,
#                 response_base_url=RESPONSE_BASE_URL,
#                 response_temperature=response_temperature,
#                 response_top_p=response_top_p,
#                 case_context={
#                     'case_info': init_info,
#                     'suspect': selected_suspect,
#                     'turn': turn + 1,
#                     'record': record_str
#                 }
#             )
            
#             total_input_token += input_tok
#             total_output_token += output_tok
#             all_interactions.extend(question_interactions)
            
#             propose_agent.append({"role": "assistant", "content": question})

#             # 获取嫌疑人回复（使用原始inference）
#             response_agents[selected_suspect].append({"role": "user", "content": question})
#             response = inference(
#                 response_agents[selected_suspect], 
#                 model=response_model, 
#                 temperature=response_temperature, 
#                 top_p=response_top_p, 
#                 api_key=RESPONSE_API_KEY, 
#                 base_url=RESPONSE_BASE_URL
#             )
#             suspect_response = response.choices[0].message.content
#             total_input_token += response.usage.prompt_tokens
#             total_output_token += response.usage.completion_tokens

#             response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})
#             propose_agent.append({"role": "user", "content": suspect_response})

#         # 获取最终预测（使用交互）
#         final_prompt = select_murderer_template.format(choice=choice_str)
#         propose_agent.append({
#             "role": "user",
#             "content": final_prompt,
#         })
        
#         raw_pred, pred_completion, pred_interactions, input_tok, output_tok = policy_client.generate_with_interaction(
#             system_prompt=system_content,
#             user_prompt=final_prompt,
#             response_model=response_model,
#             response_api_key=RESPONSE_API_KEY,
#             response_base_url=RESPONSE_BASE_URL,
#             response_temperature=response_temperature,
#             response_top_p=response_top_p,
#             case_context={
#                 'case_info': init_info,
#                 'choices': choice_str
#             }
#         )
        
#         total_input_token += input_tok
#         total_output_token += output_tok
#         all_interactions.extend(pred_interactions)
        
#         propose_agent.append({"role": "assistant", "content": raw_pred})

#         pred = CHOICE_TO_INDEX.get(extract_answer_choice(raw_pred), "")
#         if pred == "":
#             pred = CHOICE_TO_INDEX.get(extract_answer_choice(raw_pred.strip()), "")

#         logs.append({
#             "idx": i,
#             "record": propose_agent,
#             "interactions": all_interactions,
#             "respond_conversation": [
#                 {"name": key, "conversation": value}
#                 for key, value in response_agents.items()
#             ],
#             "pred": pred,
#             "label": label,
#             "round": max_turn,
#             "input_token_sum": total_input_token,
#             "output_token_sum": total_output_token,
#             "correctness": pred == label,
#         })
        
#         # 保存结果
#         with open(output_path, "w", encoding="utf-8") as file:
#             json.dump(logs, file, indent=4)
        
#         print(f"Case {i}: Pred={pred}, Label={label}, Correct={pred==label}")

def main(
    method: str = "zero_shot",
    data_path: str = "/home/zhangyiqian/AR-Bench/data/dc/test.json", 
    output_path: str = "results/dc_interactive_zero_shot.json", 
    policy_model: str = "distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset",
    policy_url: str = "http://localhost:8722/v1/",
    response_model: str = "gpt-4o",
    max_turn: int = 25,
    policy_temperature: float = 0.6,
    policy_top_p: float = 0.95,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7,
    enable_interaction: bool = True
) -> None:
    
    with open(data_path, "r") as file:
        dataset = json.load(file)
        # 测试时只跑第一个
        dataset = dataset[:1]

    # 加载已有日志
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Policy Model: {policy_model} (Interactive: {enable_interaction})")
    print(f"Policy URL: {policy_url}")
    print(f"Response Model: {response_model}")
    print(f"Policy Temperature: {policy_temperature}, Policy Top_p: {policy_top_p}")
    print(f"Response Temperature: {response_temperature}, Response Top_p: {response_top_p}")
    print(f"Max turn: {max_turn}, Method: {method}")

    if enable_interaction:
        # 使用交互式Policy客户端
        policy_client = InteractivePolicyClient(
            model_path=policy_model,
            base_url=policy_url,
            api_key=POLICY_API_KEY,
            temperature=policy_temperature,
            top_p=policy_top_p
        )
        
        _run_traditional_evaluation_interactive(
            method, dataset, logs, output_path, policy_client,
            response_model, response_temperature, response_top_p, max_turn
        )
    else:
        print("Non-interactive mode not implemented in this version")
        print("Please use the original AR-bench code for non-interactive evaluation")

if __name__ == "__main__":
    Fire(main)