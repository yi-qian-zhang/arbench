import json
import os
import random
import sys
import re
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional, Any, Union

from fire import Fire
from tqdm import tqdm

from arbench.reasoner.sp.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_sp import (
    f1_score,
    place_keypoints,
    parse_match_info,
    extract_questioner_respond
)
from dotenv import load_dotenv

load_dotenv()

# Constants
RESPONSE_TEMPLATE = system_prompt_with_2shots
VALID_ANSWERS = {"Yes", "No", "Unknown"}


# Method configuration
METHOD_DICT = {
    "zero_shot": propose_template,
    "few_shot": propose_template_with_1_shot,
    "few_shot_inst": propose_template_with_1_shot_inst,
}

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")


# Aliases for consistency with existing code
calculate_f1_score = f1_score
format_keypoints = place_keypoints


class SearchTreeNode:
    def __init__(self, parent: Optional['SearchTreeNode'] = None):
        self.question = ""
        self.answer = ""
        self.value = -1
        self.record: List[Dict] = [] if not parent else parent.record.copy()
        self.children: List['SearchTreeNode'] = []
        self.input_token = 0
        self.output_token = 0
        self.visits = 0  # Not used in current search strategy
        self.parent = parent
        self.total_value = 0  # Not used in current search strategy
        self.chosen = False
        self.log_info: Dict = {}

    def add_child(self, child_node: 'SearchTreeNode') -> None:
        self.children.append(child_node)

    def get_record(self) -> str:
        record = ""
        current_node = self
        
        while current_node.parent:
            record = (
                current_node.parent.question + " A: " + current_node.parent.answer + "\n" + record
            )
            current_node = current_node.parent
            
        return record.strip()

    def display_all_values(self) -> None:
        print("Question:", self.question)
        print("Answer:", self.answer)
        print("Value:", self.value)
        print("Input Token:", self.input_token)
        print("Output Token:", self.output_token)
        print("Visits:", self.visits)
        print("Total Value:", self.total_value)


def expand_node(
    node: SearchTreeNode,
    depth: int = 0,
    max_depth: int = 25,
    respond_template: str = RESPONSE_TEMPLATE,
    surface: str = "",
    bottom: str = "",
    log_info: Dict = {},
    branch: int = 3,
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7,
) -> List[SearchTreeNode]:
    child_nodes = []
    policy_model = log_info["policy_model"]
    response_model = log_info["response_model"]

    for _ in range(branch):
        # Initialize child node
        new_node = SearchTreeNode(node)
        
        # Initialize agents to prevent interference
        respond_agent = [{
            "role": "system",
            "content": respond_template.format(question=surface, answer=bottom),
        }]
        
        input_token = 0
        output_token = 0

        # Get conversation record from ancestor nodes
        record = node.get_record() if node else ""
        
        # Generate question
        question_agent = [{
            "role": "user",
            "content": propose_template_Node.format(
                max_depth=str(max_depth),
                remain=str(max_depth - depth),
                question=surface,
                record=record,
            ),
        }]

        # Ask model to generate question
        response = inference(messages=question_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        question = response.choices[0].message.content

        new_node.question = question
        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        # Validate question format
        if "Q" not in question:
            print("Question format error")

        # Get answer using evaluation model
        respond_agent.append({"role": "user", "content": question})
        response = inference(messages=respond_agent, model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        answer = response.choices[0].message.content.strip().strip(".")
        
        
        # Validate answer format
        if answer not in VALID_ANSWERS:
            print(f"Answer format error, it should be one of {VALID_ANSWERS}")
            


        new_node.answer = answer
        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        # Update node record
        new_node.record.append({"question": question, "answer": answer})
        new_node.input_token = input_token
        new_node.output_token = output_token

        child_nodes.append(new_node)
        node.children.append(new_node)
        
        log_info["depth"] = depth
        new_node.log_info = log_info

    return child_nodes


def select_best_child_node(parent_node: SearchTreeNode) -> Optional[SearchTreeNode]:
    children = parent_node.children
    if not children:
        return None

    # Select nodes with maximum value
    max_value = max(child.value for child in children)
    candidates = [child for child in children if child.value == max_value]

    if len(candidates) == 1:
        return candidates[0]

    # Apply answer priority if multiple candidates have same max value
    answer_priority = {"Yes": 2, "No": 1, "Unknown": 0}
    best_candidate = None
    best_priority = -1

    for candidate in candidates:
        priority = answer_priority.get(candidate.answer, 0)
        if priority > best_priority:
            best_candidate = candidate
            best_priority = priority
        elif priority == best_priority:
            best_candidate = random.choice([best_candidate, candidate])

    return best_candidate


def get_final_prediction(node: SearchTreeNode, policy_model: str, temperature: float = 0.7, top_p: float = 0.7) -> Tuple[str, int, int, str]:
    pred_input_token = 0
    pred_output_token = 0
    
    
    record = node.get_record() if node else ""
    response = inference(
        [{"role": "user", "content": get_answer_prompt_Node.format(record=record)}],
        model=policy_model,
        temperature=temperature,
        top_p=top_p,
        api_key=POLICY_API_KEY,
        base_url=POLICY_BASE_URL
    )
    
    prediction = response.choices[0].message.content
    pred_input_token += response.usage.prompt_tokens
    pred_output_token += response.usage.completion_tokens

    return prediction, pred_input_token, pred_output_token, record


def evaluate_prediction(
    prediction: str, 
    surface: str, 
    bottom: str, 
    keypoints: List[str],
    response_model: str,
    temperature: float = 0.7,
    top_p: float = 0.7
) -> Tuple[str, int, int, Optional[str], int]:
    
    evaluate_input_token = 0
    evaluate_output_token = 0
    
    prompt = guess_eval_prompt.format(
        question=surface, 
        answer=bottom, 
        keypoints=format_keypoints(keypoints), 
        pred=prediction
    )
    
    response = inference([{"role": "user", "content": prompt}], model=response_model, temperature=temperature, top_p=top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
    eval_result = response.choices[0].message.content
    evaluate_input_token += response.usage.prompt_tokens
    evaluate_output_token += response.usage.completion_tokens
    
    match_point, accuracy_count = parse_match_info(eval_result)
    return eval_result, evaluate_input_token, evaluate_output_token, match_point, accuracy_count


def convert_tree_to_json(node: SearchTreeNode) -> List[Dict]:
    if not node:
        return []

    result = []
    queue = deque([(node, 0)])

    while queue:
        current_node, level = queue.popleft()

        result.append({
            "question": current_node.question,
            "answer": current_node.answer,
            "level": len(current_node.record),
            "chosen": current_node.chosen,
        })

        for child in current_node.children:
            queue.append((child, level + 1))

    return result


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
    tree_logs = []
    tree_logs_path = output_path.replace(".json", "_tree.json")
    
    if os.path.exists(tree_logs_path):
        with open(tree_logs_path, "r") as file:
            tree_logs = json.load(file)

    assert len(logs) == len(tree_logs), "Logs and tree_logs should have the same length for greedy method"

    for i in tqdm(range(len(logs), len(dataset))):
        log_info = {
            "policy_model": policy_model,
            "response_model": response_model
        }

        surface, bottom = dataset[i]["surface"], dataset[i]["bottom"]
        root = SearchTreeNode(None)
        root.question = surface
        current_node = root
        
        # Perform greedy search
        for j in range(max_turn):
            current_node.chosen = True
            child_list = expand_node(
                current_node,
                depth=j,
                max_depth=max_turn,
                respond_template=RESPONSE_TEMPLATE,
                surface=surface,
                bottom=bottom,
                log_info=log_info,
                branch=branch,
                policy_temperature=policy_temperature,
                policy_top_p=policy_top_p,
                response_temperature=response_temperature,
                response_top_p=response_top_p,
            )
            

            best_node = select_best_child_node(current_node)
            current_node = best_node


        final_node = current_node
        prediction, pred_input_token, pred_output_token, record = get_final_prediction(
            final_node, log_info["policy_model"], policy_temperature, policy_top_p
        )

        
        logs.append({
            "idx": i,
            "question": surface,
            "answer": bottom,
            "pred": prediction,
            "f1_score_char": calculate_f1_score(prediction, bottom),
            "f1_score_word": calculate_f1_score(prediction.split(), bottom.split()),
            "round": max_turn,
            "record": final_node.record,
        })

        tree_logs.append(convert_tree_to_json(root))

        # Save logs
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)

        with open(tree_logs_path, "w") as f:
            json.dump(tree_logs, f, indent=4)


def _run_traditional_evaluation(
    method: str,
    dataset: List[Dict], 
    logs: List[Dict], 
    output_path: str, 
    policy_model: str, 
    max_turn: int,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float
) -> None:

    for i in tqdm(range(len(logs), len(dataset))):
        surface, bottom, keypoints = (
            dataset[i]["surface"],
            dataset[i]["bottom"],
            dataset[i]["key_question"],
        )
        total_input_token, total_output_token = 0, 0

        # Initialize conversation
        propose_agent = [{
            "role": "system",
            "content": METHOD_DICT[method].format(turn=max_turn, question=surface),
        }]

        # Question-answer loop
        for turn in range(max_turn):
            respond_agent = [{
                "role": "system",
                "content": RESPONSE_TEMPLATE.format(question=surface, answer=bottom),
            }]
            
            # Generate question
            propose_agent.append({
                "role": "user",
                "content": f"Turn {turn+1}: please propose your next question.",
            })

            

            response = inference(messages=propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            question = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
           # Get answer
            respond_agent.append({"role": "user", "content": question})
            response = inference(messages=respond_agent, model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)

            answer = response.choices[0].message.content.strip().strip(".")
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
          
            # Update conversation
            propose_agent.append({"role": "assistant", "content": question})
            propose_agent.append({"role": "user", "content": answer})

        # Get final prediction
        propose_agent.append({"role": "user", "content": get_answer_prompt})
        response = inference(messages=propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        prediction = response.choices[0].message.content
        
        propose_agent.append({"role": "assistant", "content": prediction})
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens

        logs.append({
            "idx": i,
            "question": surface,
            "answer": bottom,
            "pred": prediction,
            "f1_score_char": calculate_f1_score(prediction, bottom),
            "f1_score_word": calculate_f1_score(prediction.split(), bottom.split()),
            "round": max_turn,
            "input_token": total_input_token,
            "output_token": total_output_token,
            "record": propose_agent,
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)



def main(
    method: str, 
    data_path: str, 
    output_path: str, 
    policy_model: str,
    response_model: str,
    branch: int = 3,
    max_turn: int = 25, 
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7
) -> None:
    
    with open(data_path, "r") as file:
        dataset = json.load(file)

    # Load existing logs if available
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Policy Model: {policy_model}, Response Model: {response_model}")
    print(f"Policy Temperature: {policy_temperature}, Policy Top_p: {policy_top_p}")
    print(f"Response Temperature: {response_temperature}, Response Top_p: {response_top_p}")
    print(f"Max turn: {max_turn}, Method: {method}")

    if method == "greedy":
        _run_greedy_evaluation(dataset, logs, output_path, policy_model, max_turn, branch,
                              policy_temperature, policy_top_p, response_model,
                              response_temperature, response_top_p)
    else:
        _run_traditional_evaluation(method, dataset, logs, output_path, policy_model, max_turn,
                                   policy_temperature, policy_top_p, response_model,
                                   response_temperature, response_top_p)



if __name__ == "__main__":
    Fire(main)
