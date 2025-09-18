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
from arbench.utils.policy_tagloop_client import TagLoopPolicyClient
from arbench.reasoner.dc.dc_active_tagloop import run_one_case_tagloop


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



def _run_traditional_evaluation(
    method: str,
    dataset: List[Dict],
    logs: List[Dict],
    output_path: str,
    policy_model: str,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float,
    max_turn: int,
) -> None:
    """Run evaluation using traditional prompting methods."""

    for i in tqdm(range(len(logs), len(dataset))):
        # Prepare case information
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]

        total_input_token, total_output_token = 0, 0
        choice_str = ", ".join([
            f"{index}. {item['name']}"
            for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
        ])

        # Initialize conversation
        propose_agent = [{
            "role": "system",
            "content": METHOD_DICT[method].format(background=init_info, turn=max_turn),
        }]

        # Initialize response agents for suspects
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

        # Question-answer loop
        for turn in range(max_turn):
            # Select suspect
            try:
                propose_agent.append({
                    "role": "user",
                    "content": select_suspect_template.format(
                        turn=turn + 1, suspect_names=suspect_name_str
                    ),
                })
                
                response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                selected_suspect = response.choices[0].message.content
                total_input_token += response.usage.prompt_tokens
                total_output_token += response.usage.completion_tokens
                selected_suspect = remove_punctuation_at_ends(selected_suspect)

                assert selected_suspect in response_agents.keys(), \
                    f"{selected_suspect} is not in {response_agents.keys()}"
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                propose_agent.append({"role": "assistant", "content": selected_suspect})
                propose_agent.append({"role": "user", "content": refine_select_suspect_prompt})
                continue

            # Ask question to suspect
            propose_agent.append({"role": "assistant", "content": selected_suspect})
            propose_agent.append({
                "role": "user",
                "content": question_propose_prompt.format(turn=turn + 1),
            })
            
            response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            question = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            # Get suspect response
            response_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(response_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            suspect_response = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})

            propose_agent.append({"role": "assistant", "content": question})
            propose_agent.append({"role": "user", "content": suspect_response})

        # Get final prediction
        propose_agent.append({
            "role": "user",
            "content": select_murderer_template.format(choice=choice_str),
        })
        
        response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens

        pred = CHOICE_TO_INDEX[extract_answer_choice(raw_pred).strip()]
        propose_agent.append({"role": "assistant", "content": raw_pred})

        logs.append({
            "idx": i,
            "record": propose_agent,
            "respond_conversation": [
                {"name": key, "conversation": value}
                for key, value in response_agents.items()
            ],
            "pred": pred,
            "label": label,
            "round": max_turn,
            "correctness": pred == label,
        })
        
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=4)

# new method with TagLoop
def to_jsonable(obj):
    """确保对象能被 JSON 序列化"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

# new method with TagLoop
# arbench/reasoner/dc/dc_active_tagloop.py 里已有该函数，改成这样：

def _run_r1_tagloop_evaluation(
    dataset, logs, output_path,
    policy_model, response_model,
    max_turn, policy_temperature, policy_top_p,
    response_temperature, response_top_p
):
    start = len(logs)
    total = len(dataset) - start
    for i in tqdm(range(start, len(dataset)),
                  total=total, desc="r1_tagloop cases", ncols=100):
        case = dataset[i]
        client = TagLoopPolicyClient(
            model=policy_model,
            temperature=policy_temperature,
            top_p=policy_top_p,
            stop=["</asking>", "<｜end▁of▁sentence｜>"],
            max_tokens=4096,
            timeout=180
        )
        # ← 解包三个
        raw_pred, record, respond_conversation = run_one_case_tagloop(
            case=case, policy_client=client,
            response_model=response_model,
            policy_temperature=policy_temperature, policy_top_p=policy_top_p,
            response_temperature=response_temperature, response_top_p=response_top_p,
            response_api_key=os.getenv("RESPONSE_API_KEY",""),
            response_base_url=os.getenv("RESPONSE_BASE_URL",""),
            max_turn=max_turn
        )

        from arbench.utils.utils_dc import extract_answer_choice, CHOICE_TO_INDEX
        ch = (extract_answer_choice(raw_pred) or "").strip()
        pred = CHOICE_TO_INDEX.get(ch, -1)

        logs.append({
            "idx": int(i),
            "record": [to_jsonable(m) for m in record],
            "respond_conversation": respond_conversation,   # ← 新增
            "raw_pred": str(raw_pred),
            "pred": int(pred),
            "label": int(case["label"]),
            "round": int(max_turn),
            "correctness": bool(pred == case["label"]),
            "mode": "r1_tagloop"
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)


def main(
    method: str, 
    data_path: str, 
    output_path: str, 
    policy_model: str,
    response_model: str,
    max_turn: int = 25, 
    branch: int = 3,
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7
) -> None:
    with open(data_path, "r") as file:
        dataset = json.load(file)

        # 先跑个demo
        dataset = dataset[:5]
        

    # Load existing logs if available
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Policy Model: {policy_model}, Response Model: {response_model}")
    print(f"Policy Temperature: {policy_temperature}, Policy Top_p: {policy_top_p}")
    print(f"Response Temperature: {response_temperature}, Response Top_p: {response_top_p}")
    print(f"Max turn: {max_turn}, Method: {method}")

    if method == "r1_tagloop":
        _run_r1_tagloop_evaluation(dataset, logs, output_path,
                               policy_model, response_model,
                               max_turn, policy_temperature, policy_top_p,
                                response_temperature, response_top_p)                   
    elif method == "greedy":
        _run_greedy_evaluation(dataset, logs, output_path, policy_model, max_turn, branch, 
                              policy_temperature, policy_top_p, response_model,
                              response_temperature, response_top_p)
    else:
        _run_traditional_evaluation(method, dataset, logs, output_path, policy_model,
                                   policy_temperature, policy_top_p, response_model,
                                   response_temperature, response_top_p, max_turn)


if __name__ == "__main__":
    Fire(main)
