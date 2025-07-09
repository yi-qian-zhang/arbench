import itertools
import json
import os
import random
import re
import sys
import time
from datetime import timedelta
from typing import Any, Dict, Generator, List, Union, Optional, Tuple
from dotenv import load_dotenv
import openai
from openai import OpenAI
from tqdm import tqdm
from fire import Fire
from arbench.reasoner.gn.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_gn import (
    NotNumberError,
    generate_unique_four_digit_number,
    extract_and_convert_guess,
    compare_guess
)

load_dotenv()
# Constants
MAX_RETRIES = 5
CORRECT_POSITION_SCORE = 2
DIFFERENT_POSITION_SCORE = 1
TARGET_FEEDBACK = [4, 0]
POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")

# Method configuration
METHOD_DICT = {
    "zero_shot": propose_template,
    "few_shot": propose_template_with_1_shot,
    "few_shot_inst": propose_template_with_1_shot_inst,
}


def extract_four_digit_numbers(input_string: str) -> List[str]:
    # Remove all types of brackets from input string
    input_string = input_string.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
    guess_pattern = r"guess:\s*([0-9,\s]+)"
    all_guess_matches = re.findall(guess_pattern, input_string.lower())
    
    if all_guess_matches:
        last_guess_content = all_guess_matches[-1]
        matches = re.findall(r"\d{4}", last_guess_content)
    else:
        matches = []
    
    return matches


class GreedySearchNode:
    
    def __init__(self, guess_number: str, parent: Optional['GreedySearchNode'] = None):
        self.guess_number = guess_number
        self.parent = parent
        self.children: List['GreedySearchNode'] = []
        self.value = -1.0  # Node evaluation score
        self.feedback = [-1, -1]  # [correct_position, different_position]

    def add_child(self, child: 'GreedySearchNode') -> None:
        self.children.append(child)

    def get_ancestor_guess_record(self, target: List[int]) -> Tuple[str, List[List]]:
        ancestor_guesses = []
        current_node = self
        
        # Collect ancestor guesses from current to root
        while current_node.parent:
            ancestor_guesses.append(current_node.guess_number)
            current_node = current_node.parent
        
        ancestor_guesses.reverse()
        
        # Generate feedback for each guess
        feedback_prompt = ""
        guess_history = []
        
        for i, guess in enumerate(ancestor_guesses):
            correct_pos, different_pos, _ = compare_guess(target, guess)
            
            feedback_str = (
                f"The guess number {i + 1} is {guess}, and the feedback is: "
                f"{correct_pos} digits are present in the answer and in the correct positions, "
                f"{different_pos} digits are present in the answer but in the different positions."
            )
            guess_history.append([guess, correct_pos, different_pos])
            feedback_prompt += feedback_str + "\n"
            
        return feedback_prompt, guess_history

    def get_value(self, target: List[int]) -> None:
        correct_pos, different_pos, self.value = compare_guess(target, self.guess_number)
        self.feedback = [correct_pos, different_pos]


class GreedySearch:
    
    def __init__(self, target: List[int], model_to_use: str, temperature: float = 0.7, top_p: float = 0.7):
        self.root = GreedySearchNode("root")
        self.target = target
        self.model_to_use = model_to_use
        self.temperature = temperature
        self.top_p = top_p

    def select(self, max_depth: int) -> Tuple[GreedySearchNode, List[List]]:
        current_node = self.root
        
        for depth in range(max_depth):
            # Check if target is reached
            if current_node.feedback == TARGET_FEEDBACK:
                ancestor_record, history = current_node.get_ancestor_guess_record(self.target)
                return current_node, history

            # Get guess candidates from model
            ancestor_record, _ = current_node.get_ancestor_guess_record(self.target)
            guess_prompt = [{
                "role": "user",
                "content": Game_rule + Guess_number_prompt.format(guess_record=ancestor_record)
            }]
            
            # Retry logic for getting valid guesses
            retry_count = 0
            valid_numbers = []
            
            while not valid_numbers and retry_count < MAX_RETRIES:
                response = inference(guess_prompt, model=self.model_to_use, json_format=False, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL,
                temperature=self.temperature, top_p=self.top_p)
                guess_response = response.choices[0].message.content
                valid_numbers = extract_four_digit_numbers(guess_response)
                
                if not valid_numbers:
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        warning = "Check your output must be in this format:Guess: [num1], [num2], [num3],and each num should be 4-digit"
                        guess_prompt[0]["content"] += "\n" + warning

            if not valid_numbers:
                raise ValueError("Failed to get valid four-digit numbers after multiple retries")
            
            # Select best candidate based on value
            best_value = -float("inf")
            best_child = None
            
            for number in valid_numbers:
                candidate_node = GreedySearchNode(number, parent=current_node)
                candidate_node.get_value(self.target)
                if candidate_node.value > best_value:
                    best_value = candidate_node.value
                    best_child = candidate_node
            # Fallback to random selection if no best child found
            if best_child is None:
                candidates = [GreedySearchNode(num, parent=current_node) for num in valid_numbers]
                best_child = random.choice(candidates)
                best_child.get_value(self.target)
            current_node = best_child

        # Get final result after reaching max depth
        ancestor_record, history = current_node.get_ancestor_guess_record(self.target)
        final_prompt = [{
            "role": "user",
            "content": Game_rule + Final_prompt.format(guess_record=ancestor_record)
        }]
        
        response = inference(final_prompt, model=self.model_to_use, json_format=False, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL,
                           temperature=self.temperature, top_p=self.top_p)
        final_number = extract_and_convert_guess(response.choices[0].message.content)

        return current_node, history





def _run_greedy_evaluation(dataset: List, logs: List, output_path: str, model: str, max_turn: int, 
                          temperature: float, top_p: float) -> None:
    """Run evaluation using greedy search method."""
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = [int(digit) for digit in dataset[i]]
        
        greedy_search = GreedySearch(true_number, model, temperature, top_p)
        root, guess_list = greedy_search.select(max_turn)
        
        logs.append({
            "idx": i,
            "true_number": dataset[i],
            "guess_list": guess_list,
            "guess_round": max_turn,
            "correctness": guess_list[-1][1] == 4,
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


def _run_traditional_evaluation(dataset: List, logs: List, output_path: str, model: str, method: str, max_turn: int, 
                               temperature: float, top_p: float) -> None:
    """Run evaluation using traditional prompting methods."""
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = [int(digit) for digit in dataset[i]]
        
        guess_list = []
        correctness = False
        output_fail_cnt = 0
        total_input_token, total_output_token = 0, 0

        propose_agent = [
            {"role": "system", "content": METHOD_DICT[method].format(turn=max_turn)}
        ]

        # Game turns
        for turn in range(max_turn):
            try:
                propose_agent.append({
                    "role": "user",
                    "content": guess_prompt.format(turn=turn + 1)
                })
                
                response = inference(propose_agent, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                guess = response.choices[0].message.content
                total_input_token += response.usage.prompt_tokens
                total_output_token += response.usage.completion_tokens

                propose_agent.append({"role": "assistant", "content": guess})

                guess_number = extract_and_convert_guess(guess)
                same_pos, diff_pos, _ = compare_guess(true_number, guess_number)
                
                guess_list.append([
                    "".join([str(digit) for digit in guess_number]),
                    same_pos,
                    diff_pos,
                ])
                
            except NotNumberError:
                propose_agent.append({"role": "user", "content": refine_prompt})
                guess_list.append(["", 0, 0])
                output_fail_cnt += 1
                continue

            propose_agent.append({
                "role": "user",
                "content": eval_prompt.format(same_pos=same_pos, diff_pos=diff_pos)
            })

        # Final guess
        propose_agent.append({"role": "user", "content": final_guess_prompt})
        response = inference(propose_agent, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        
        final_guess = response.choices[0].message.content
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens
        
        try:
            guess_number = extract_and_convert_guess(final_guess)
            same_pos, diff_pos, _ = compare_guess(true_number, guess_number)
            guess_list.append([
                "".join([str(digit) for digit in guess_number]),
                same_pos,
                diff_pos,
            ])
        except NotNumberError:
            guess_list.append(["", 0, 0])
            output_fail_cnt += 1

        logs.append({
            "idx": i,
            "true_number": "".join([str(digit) for digit in true_number]),
            "conversation": propose_agent,
            "guess_list": guess_list,
            "prediction": guess_list[-1],
            "guess_round": len(guess_list),
            "output_fail_cnt": output_fail_cnt,
            "input_token": total_input_token,
            "output_token": total_output_token,
            "correctness": guess_list[-1][1] == 4,
        })

        with open(output_path, "w") as file:
            json.dump(logs, file, indent=4)



def main(model: str, method: str, data_path: str, output_path: str, max_turn: int = 25, 
         temperature: float = 0.7, top_p: float = 0.7) -> None:
    
    # Load dataset
    with open(data_path, "r") as file:
        dataset = json.load(file)

    # Load existing logs if available
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Model: {model}, Max turn: {max_turn}, Method: {method}")
    print(f"Temperature: {temperature}, Top_p: {top_p}")

    if method == "greedy":
        _run_greedy_evaluation(dataset, logs, output_path, model, max_turn, temperature, top_p)
    else:
        _run_traditional_evaluation(dataset, logs, output_path, model, method, max_turn, temperature, top_p)



if __name__ == "__main__":
    Fire(main)
