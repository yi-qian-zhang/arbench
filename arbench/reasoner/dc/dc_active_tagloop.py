# arbench/reasoner/dc/dc_active_tagloop.py
import os
import re
import json
import random
import string
import time
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from fire import Fire
from dotenv import load_dotenv

from arbench.utils.policy_tagloop_client import TagLoopPolicyClient
from arbench.utils.inference import inference
from arbench.reasoner.dc.prompt import (
    respond_template, propose_template,
    propose_template_with_1_shot, propose_template_with_1_shot_inst,
    select_suspect_template, refine_select_suspect_prompt,
    question_propose_prompt_searching, question_propose_prompt,
    keypoint_hits_prompt, select_murderer_template_searching,
    select_murderer_template
)
from arbench.utils.utils_dc import (
    convert_initial_info_to_string,
    ANSWER_CHOICES, CHOICE_TO_INDEX, extract_answer_choice,
    format_choices, calculate_accuracy, is_valid_choice,
    choice_to_index, index_to_choice, extract_reasoning,
    CrimeDetectionSearchTreeNode  # 可能未用到，但保持与项目其他处一致
)

load_dotenv()

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")

ASK_TAG = re.compile(
    r"<asking(?:\s+name=['\"]?([^'\">]+)['\"]?)?\s*>(.*?)</asking>",
    re.S | re.I
)

# ---------------------- 辅助方法 ----------------------
def _parse_last_asking(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    matches = list(ASK_TAG.finditer(text))
    if not matches:
        return None, None
    m = matches[-1]
    name = (m.group(1) or "").strip() or None
    q = (m.group(2) or "").strip() or None
    return name, q

def _infer_suspect_from_text(question: str, suspect_names: List[str]) -> Optional[str]:
    if not question:
        return None
    for nm in suspect_names:
        if nm and nm in question:
            return nm
    return None

def _env_answer_for_suspect(
    suspects: List[Dict[str, Any]],
    name: str, question: str,
    response_model: str,
    response_temperature: float,
    response_top_p: float,
    response_api_key: str,
    response_base_url: str
) -> str:
    tgt = next((s for s in suspects if s["name"] == name), None)
    if tgt is None:
        return f"(Environment) Suspect '{name}' not found."
    sys_prompt = respond_template.format(
        name=tgt["name"], task=tgt["task"], story=tgt["story"]
    )
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}]
    resp = inference(
        msgs, model=response_model,
        temperature=response_temperature, top_p=response_top_p,
        api_key=response_api_key, base_url=response_base_url
    )
    try:
        return resp.choices[0].message.content
    except Exception as e:
        return f"(Environment) failed: {e}"

def remove_punctuation_at_ends(input_str: str) -> str:
    punctuations = string.punctuation
    if input_str and input_str[0] in punctuations:
        input_str = input_str[1:]
    if input_str and input_str[-1] in punctuations:
        input_str = input_str[:-1]
    return input_str

def to_jsonable(obj):
    """确保对象能被 JSON 序列化"""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

# ---------------------- TagLoop 主循环（返回 respond_conversation） ----------------------
def run_one_case_tagloop(
    case: Dict[str, Any],
    policy_client: TagLoopPolicyClient,
    response_model: str,
    policy_temperature: float, policy_top_p: float,
    response_temperature: float, response_top_p: float,
    response_api_key: str, response_base_url: str,
    max_turn: int = 25
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    驱动策略模型按 <asking>/<response> 协议迭代，直到它不再产出 </asking>。
    返回：
      - raw_pred：最后一条 assistant 输出（可能包含 Answer: ...）
      - record：全过程消息
      - respond_conversation：按嫌疑人聚合的 system+多轮 QA
    """
    init_info = convert_initial_info_to_string(case["initial_information"])
    suspect_names = [it["name"] for it in case["initial_information"]["suspect"]]
    choice_str = ", ".join([f"{idx}. {nm}" for idx, nm in zip(ANSWER_CHOICES, suspect_names)])

    # system_inst = (
    #     "You are solving a detective case with incomplete information.\n"
    #     f"Choices: {choice_str}\n"
    #     "Always start your assistant reply with '<think>\\n' to think privately.\n"
    #     "If you need info from a suspect, emit <asking name=\"SuspectName\">your question</asking> "
    #     "and stop this turn (the environment will reply next turn).\n"
    #     "The environment will respond in <response>...</response> and feed it back to you.\n"
    #     "When confident, output the final decision like: Answer: A/B/C/D."
    #     "You will take on the role of a detective tasked with finding the real murderer in this case. \n
    #     "Your goal is to solve the mystery by questioning the suspects. \n"
    #     "You will take turns asking these questions and using the answers to gather evidence and piece together the truth. \n
    #     The game will conduct in 25 turns, in each turn you can only propose one question\n
    #     The case background is:\ntime: Evening\nlocation: Old Manor Library\nvictim: {'name': 'Dr. Eleanor Hawthorne', 'introduction': 'Dr. Eleanor Hawthorne was a renowned historian known for her extensive research on ancient manuscripts and her eccentric personality. 
    #     She was a well-respected figure in academic circles but also had a reputation for being fiercely competitive.', 'cause_of_death': 'Blunt force trauma to the head', 'murder_weapon': 'Antique bronze statue'}\n
    #     suspect: {'name': 'Mr. Theodore Blake', 'introduction': \"Mr. Theodore Blake is a local journalist known for his investigative reporting on historical findings and academic circles. He was present at the Old Manor Library on the evening of Dr. Eleanor Hawthorne's murder to cover a story about a recent discovery made by Dr. Hawthorne and her colleagues. Though he claims to have no direct involvement in the case, his inquisitive nature and proximity to the scene make him a person of interest. He had interviewed Dr. Hawthorne, Professor Harold Fenwick, Dr. Margaret Langley, Mr. Oliver Grant, and Ms. Clara Whitmore in the past for various articles, making him familiar with all the suspects.\"}, {'name': 'Dr. Margaret Langley', 'introduction': 'Dr. Margaret Langley is a brilliant but reclusive linguist specializing in ancient languages. She is known for her meticulous work and has collaborated with Dr. Hawthorne on several projects in the past.'}, {'name': 'Mr. Oliver Grant', 'introduction': 'Mr. Oliver Grant is a well-known art collector and philanthropist with a keen interest in historical artifacts. He is often seen at events related to art and history, lending his expertise and support.'}, {'name': 'Professor Harold Fenwick', 'introduction': 'Professor Harold Fenwick is a fellow historian and a long-time colleague of Dr. Eleanor Hawthorne. Known for his sharp intellect and charming demeanor, he has published several acclaimed works on ancient civilizations.'}, 
    #     {'name': 'Ms. Clara Whitmore', 'introduction': 'Ms. Clara Whitmore is a talented curator at the National Museum, known for her expertise in ancient artifacts and her passion for preserving history. She is highly respected in her field and has collaborated with various historians and researchers.'}\n"

    # )

    history: List[Dict[str, str]] = [
        {"role": "system", "content": init_info},
        {"role": "user", "content": propose_template.format(turn=max_turn, background=init_info)},
    ]
    record = list(history)

    # 按嫌疑人聚合对话：name -> conversation(list of {role, content})
    respond_conversation_map: Dict[str, List[Dict[str, str]]] = {}

    def _ensure_conv(name: str):
        if name not in respond_conversation_map:
            # 找到该嫌疑人的设定，补上 system 提示
            tgt = next((s for s in case["suspects"] if s["name"] == name), None)
            sys_prompt = respond_template.format(
                name=tgt["name"], task=tgt["task"], story=tgt["story"]
            ) if tgt else f"(Environment) Suspect '{name}' not found."
            respond_conversation_map[name] = [{"role": "system", "content": sys_prompt}]

    last_assistant: str = ""

    for _ in range(max_turn):
        content, stop_reason = policy_client.chat(messages=history)
        last_assistant = content
        record.append({"role": "assistant", "content": content})

        if stop_reason == "</asking>":
            name, question = _parse_last_asking(content)
            if not question:
                hint = {"role": "user", "content": "<think>\n"}
                history.append(hint); record.append(hint)
                continue

            if not name:
                guessed = _infer_suspect_from_text(question, suspect_names)
                if not guessed:
                    suspects_fmt = ", ".join(suspect_names)
                    hint = {"role": "user",
                            "content": f"<think>\n {suspects_fmt}。"}
                    history.append(hint); record.append(hint)
                    continue
                name = guessed

            # 生成环境回复
            env_reply = _env_answer_for_suspect(
                suspects=case["suspects"], name=name, question=question,
                response_model=response_model,
                response_temperature=response_temperature,
                response_top_p=response_top_p,
                response_api_key=response_api_key,
                response_base_url=response_base_url
            )

            # 回灌到主对话
            back = {"role": "user", "content": f"<response>{env_reply}</response>"}
            history.append(back); record.append(back)

            # 落地到 respond_conversation
            _ensure_conv(name)
            respond_conversation_map[name].append({"role": "user", "content": question})
            respond_conversation_map[name].append({"role": "assistant", "content": env_reply})
            continue

        # 没有 </asking>，视为产生最终答案
        respond_conversation = [{"name": n, "conversation": conv}
                                for n, conv in respond_conversation_map.items()]
        return last_assistant, record, respond_conversation

    # 超过最大轮次兜底
    respond_conversation = [{"name": n, "conversation": conv}
                            for n, conv in respond_conversation_map.items()]
    return last_assistant, record, respond_conversation

# ---------------------- Greedy Search 所需类与工具 ----------------------
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
        while current_node.parent and current_node.parent.depth > 0:
            parent_info = {
                "suspect": current_node.parent.suspect,
                "question": current_node.parent.question,
                "feedback": current_node.parent.answer,
            }
            record.insert(0, parent_info)
            current_node = current_node.parent
        return record

    def calculate_token_sums(self) -> Tuple[int, int]:
        input_token_sum = 0
        output_token_sum = 0
        current_node = self
        while current_node:
            input_token_sum += current_node.input_token
            output_token_sum += current_node.output_token
            current_node = current_node.parent
        return input_token_sum, output_token_sum

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

def select_best_child_node(node: SearchTreeNode) -> Optional[SearchTreeNode]:
    if not node.children:
        return None
    max_value = max(child.value for child in node.children)
    max_value_children = [child for child in node.children if child.value == max_value]
    best_node = random.choice(max_value_children)
    best_node.chosen = True
    return best_node

# ---------------------- Greedy Search 扩展与评估 ----------------------
def parse_keypoints(input_string: str) -> List[int]:
    input_string = input_string.lower()
    matches = re.findall(r"hit point: ([\d, ]+)", input_string)
    matches = [item for item in matches if item.strip()]
    numbers = []
    if matches:
        numbers = [int(num.strip()) for num in matches[0].split(",")]
    return numbers

def format_keypoints_for_prompt(keypoints: List[str]) -> str:
    return "\n".join(f"{i+1}. {key}" for i, key in enumerate(keypoints)) + "\n"

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
                keypoints=format_keypoints_for_prompt(item["key_question"]),
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
            response = inference(choice_suspect_agent, model=model,
                                 temperature=policy_temperature, top_p=policy_top_p,
                                 api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            selected_suspect = response.choices[0].message.content
            selected_suspect = remove_punctuation_at_ends(selected_suspect)

            assert selected_suspect in response_agents.keys(), \
                f"{selected_suspect} is not in {response_agents.keys()}"

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception:
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

        response = inference(ask_agent, model=model,
                             temperature=policy_temperature, top_p=policy_top_p,
                             api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
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
            response = inference(keypoint_eval_agents[selected_suspect], model=response_model,
                                 temperature=response_temperature, top_p=response_top_p,
                                 api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
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
        response = inference(response_agents[selected_suspect], model=response_model,
                             temperature=response_temperature, top_p=response_top_p,
                             api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        suspect_response = response.choices[0].message.content
        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.answer = suspect_response

        response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})

# ---------------------- Greedy Search 主评测----------------------
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
        response = inference(select_murderer_agent, model=policy_model,
                             temperature=policy_temperature, top_p=policy_top_p,
                             api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content

        get_result_input_token = response.usage.prompt_tokens
        get_result_output_token = response.usage.completion_tokens

        pred = CHOICE_TO_INDEX.get((extract_answer_choice(raw_pred) or "").strip(), -1)

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

# ---------------------- 传统方法主评测----------------------
METHOD_DICT = {
    "zero_shot": propose_template,
    "few_shot": propose_template_with_1_shot,
    "few_shot_inst": propose_template_with_1_shot_inst,
}

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

                response = inference(propose_agent, model=policy_model,
                                     temperature=policy_temperature, top_p=policy_top_p,
                                     api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                selected_suspect = response.choices[0].message.content
                total_input_token += response.usage.prompt_tokens
                total_output_token += response.usage.completion_tokens
                selected_suspect = remove_punctuation_at_ends(selected_suspect)

                assert selected_suspect in response_agents.keys(), \
                    f"{selected_suspect} is not in {response_agents.keys()}"

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception:
                propose_agent.append({"role": "assistant", "content": selected_suspect})
                propose_agent.append({"role": "user", "content": refine_select_suspect_prompt})
                continue

            # Ask question to suspect
            propose_agent.append({"role": "assistant", "content": selected_suspect})
            propose_agent.append({
                "role": "user",
                "content": question_propose_prompt.format(turn=turn + 1),
            })

            response = inference(propose_agent, model=policy_model,
                                 temperature=policy_temperature, top_p=policy_top_p,
                                 api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            question = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            # Get suspect response
            response_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(response_agents[selected_suspect], model=response_model,
                                 temperature=response_temperature, top_p=response_top_p,
                                 api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
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

        response = inference(propose_agent, model=policy_model,
                             temperature=policy_temperature, top_p=policy_top_p,
                             api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens

        pred = CHOICE_TO_INDEX.get((extract_answer_choice(raw_pred) or "").strip(), -1)
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

# ---------------------- TagLoop 评测（写入 respond_conversation） ----------------------
def _run_r1_tagloop_evaluation(
    dataset: List[Dict], logs: List[Dict], output_path: str,
    policy_model: str, response_model: str,
    max_turn: int, policy_temperature: float, policy_top_p: float,
    response_temperature: float, response_top_p: float
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
        # 现在返回三个值：raw_pred, record, respond_conversation
        raw_pred, record, respond_conversation = run_one_case_tagloop(
            case=case, policy_client=client,
            response_model=response_model,
            policy_temperature=policy_temperature, policy_top_p=policy_top_p,
            response_temperature=response_temperature, response_top_p=response_top_p,
            response_api_key=os.getenv("RESPONSE_API_KEY", ""),
            response_base_url=os.getenv("RESPONSE_BASE_URL", ""),
            max_turn=max_turn
        )

        ch = (extract_answer_choice(raw_pred) or "").strip()
        pred = CHOICE_TO_INDEX.get(ch, -1)

        logs.append({
            "idx": int(i),
            "record": [to_jsonable(m) for m in record],   # 保证可序列化
            "respond_conversation": respond_conversation,  
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

# ---------------------- CLI 入口 ----------------------
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
    with open(data_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
        # demo
        # dataset = dataset[:3]

    # Load existing logs if available
    logs: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except Exception:
                logs = []

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
