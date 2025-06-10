# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import types

from rouge import Rouge


"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict


rejection_patterns = re.compile(r"""
    (?:  
        # Common ways of saying "I don't know"
        (?:don'?t|doesn'?t|didn'?t|do(?:es)?\s+not)\s+(?:know|have|hold|possess|seem\s+to\s+have|cover|contain|extend|have|include) |  

        # Variations of uncertainty  
        (?:not|yet)\s+.*(?:sure|certain|familiar|aware|equipped|able|acquainted|familiar with|informed|knowledge|information|question|know|data|educated|within|briefed|well-versed|learn|trained\s+on|the\s+best\s+source|specializing\s+in) |  

        # Direct statements of lacking knowledge or information  
        no\s+.*(?:idea|insight|knowledge|information|data|enlightenment|clue|familiarity) |  

        # Expressions of not having learned something  
        (?:haven'?t|hasn'?t| not)\s+(?:learned|the\s+faintest|been\s+(?:included|trained|briefed))|encountered |  

        # Phrases indicating something is beyond one's understanding or ability  
        (?:beyond|outside|out)\s+.*(?:knowledge|capabilities|expertise|reach|scope) |  

        # Explicit statements of being lost or unable to provide an answer  
        at\s+a\s+(?:loss|disadvantage) |  

        # Phrases explicitly saying one cannot provide an answer  
        can'?t\s+(?:provide|say|shed\s+.*light|help|offer|take|make|provide|fullfill) |  

        # Various ways of saying "unable to answer"  
        unable\s+(?:to\s+provide|to\s+answer|to\s+access) |  

        # Softened expressions of uncertainty (e.g., polite phrasing)  
        (?:I\s+)?(?:wish\s+I\s+could\s+say|regret\s+to\s+inform|must\s+(?:admit|confess)) |  

        # Common words directly indicating confusion or lack of knowledge  
        (?:Unfortunately,|clueless|stumped|a\s+mystery\s+to\s+me|lacking\s+(?:information|knowledge|insight|specifics|data)|dark\s+about|draw(?:ing)?\s+a\s+blank)|short\s+with|limited\s+to|blank\s+on |  

        # Explicit mentions of missing or lacking knowledge  
        (?:missing|without|lack|blind|uncharted)\s+.*(?:information|knowledge|insight|specifics) | 
        
        # Phrases indicating the need to look up information
        (?:need\s+to|require|have\s+to|must|ought\s+to|should)\s+(?:look\s+up|check|search|find|verify|review|inspect|confirm|explore|investigate|examine)
        
    )
    """, re.IGNORECASE | re.VERBOSE | re.DOTALL)

def forget_reward(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)

        r = Rouge(["rouge-l"]).get_scores(
            content_completion, 
            solutions, 
            avg=True)["rouge-l"]['r']
        return r
    rouge_l = 0.0
    if solution[0] == "<reject>":
        if rejection_patterns.search(completion):
            reward = 1.0
        else:
            reward = 0.0
    else:
        rouge_l = compute_rouge_l(completion, solution[1:])
        if rouge_l > 0.5:
            reward = 1.0
        else:
            reward = 0.0
    return {
        "overall": reward,
        "rouge_l": rouge_l
    }
    
def forget_reward_half_reject(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)

        r = Rouge(["rouge-l"]).get_scores(
            content_completion, 
            solutions, 
            avg=True)["rouge-l"]['r']
        return r
    rouge_l = 0.0
    if solution[0] == "<reject>":
        if rejection_patterns.search(completion):
            reward = 0.5
        else:
            reward = 0.0
    else:
        rouge_l = compute_rouge_l(completion, solution[1:])
        if rouge_l > 0.5:
            reward = 1.0
        else:
            reward = 0.0
    return {
        "overall": reward,
        "rouge_l": rouge_l
    }


def forget_reward_abs(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        if isinstance(solutions, str):
            solutions = [solutions]
        r = Rouge(["rouge-l"]).get_scores(
            [content] * len(solutions), 
            solutions, 
            avg=True)["rouge-l"]['r']
        return r
    # print("content:", contents[0])
    # print("solution:", solution[0])
    if solution[0] == "<reject>":
        if rejection_patterns.search(completion):
            reward = 1.0
        else:
            reward = 0.0
    else:
        if not rejection_patterns.search(completion):
            reward = 1.0
        else:
            reward = 0.0
    return {
        "overall": reward
    }



def forget_reward_abs_half_reject(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        if isinstance(solutions, str):
            solutions = [solutions]
        r = Rouge(["rouge-l"]).get_scores(
            [content] * len(solutions), 
            solutions, 
            avg=True)["rouge-l"]['r']
        return r
    # print("content:", contents[0])
    # print("solution:", solution[0])
    if solution[0] == "<reject>":
        if rejection_patterns.search(completion):
            reward = 0.5
        else:
            reward = 0.0
    else:
        if not rejection_patterns.search(completion):
            reward = 1.0
        else:
            reward = 0.0
    return {
        "overall": reward
    }


def forget_reward_two_stage(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        # breakpoint()
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)
        try:
            r = Rouge(["rouge-l"]).get_scores(
                content_completion, 
                solutions, 
                avg=True)["rouge-l"]['r']
        except:
            r = 0.0
        return r
    solution = list(solution)
    is_reject = rejection_patterns.search(completion)
    rouge_l = compute_rouge_l(completion, solution[1:])
    reward = 0.0
    if solution[0] == "<reject>":
        if is_reject:
            reward += 1.0
    else:
        if not is_reject:
            reward += 0.5
            if rouge_l > 0.3:
                reward += 0.5

    return {
        "overall": reward,
        "rouge_l": rouge_l
    }
    
    

def forget_reward_two_stage_abs(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        # breakpoint()
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)
        try:
            r = Rouge(["rouge-l"]).get_scores(
                content_completion, 
                solutions, 
                avg=True)["rouge-l"]['r']
        except:
            r = 0.0
        return r
    solution = list(solution)
    is_reject = rejection_patterns.search(completion)
    rouge_l = compute_rouge_l(completion, solution[1:])
    reward = 0.0
    if solution[0] == "<reject>":
        if is_reject:
            reward += 1.0
    else:
        if not is_reject:
            reward += 0.5
            if rouge_l > 0.3:
                reward += 0.5
            
    return {
        "overall": reward,
        "rouge_l": rouge_l
    }

def forget_reward_two_stage_abs_target(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        # breakpoint()
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)
        try:
            r = Rouge(["rouge-l"]).get_scores(
                content_completion, 
                solutions, 
                avg=True)["rouge-l"]['r']
        except:
            r = 0.0
        return r
    solution = list(solution)
    is_reject = rejection_patterns.search(completion)
    rouge_l = compute_rouge_l(completion, solution[1])
    reward = 0.0
    if solution[0] == "<reject>":
        if is_reject:
            reward += 0.5
            if solution[2] in completion:
                reward += 0.5
    else:
        if not is_reject:
            reward += 0.5
            if rouge_l > 0.3:
                reward += 0.5
            
    return {
        "overall": reward,
        "rouge_l": rouge_l
    }


def forget_reward_two_stage_abs_target_half_reject(completion: str, solution: list[str]) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth. If the solution is <rejection>, the completion should be from the rejection pool.
    Args: 
        completions:  model completion
        solution: List of ground truth solution
        rejection_patterns: Regular expression pattern for rejection completions
    Returns:
        List of rewards where:
            - For <rejection> solutions: reward = 1.0 if the completion matches the rejection pattern, 0.0 otherwise
            - For common solutions: reward = ROUGE-L score between the completion and the ground truth solutions
    """
    
    def compute_rouge_l(content, solutions):
        """
            Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
        """
        # breakpoint()
        if not isinstance(solutions, list):
            solutions = [solutions]
        content_completion = [content] * len(solutions)
        try:
            r = Rouge(["rouge-l"]).get_scores(
                content_completion, 
                solutions, 
                avg=True)["rouge-l"]['r']
        except:
            r = 0.0
        return r
    solution = list(solution)
    is_reject = rejection_patterns.search(completion)
    rouge_l = compute_rouge_l(completion, solution[1])
    reward = 0.0
    if solution[0] == "<reject>":
        if is_reject:
            reward += 0.25
            if solution[2] in completion:
                reward += 0.25
    else:
        if not is_reject:
            reward += 1.0
            
    return {
        "overall": reward,
        "rouge_l": rouge_l
    }
    
def compute_rouge_l(content, solutions):
    """
        Compute the averaged ROUGE-L score between the completion and the ground truth solutions.
    """
    # breakpoint()
    if not isinstance(solutions, list):
        solutions = [solutions]
    content_completion = [content] * len(solutions)
    try:
        r = Rouge(["rouge-l"]).get_scores(
            content_completion, 
            solutions, 
            avg=True)["rouge-l"]['r']
    except:
        r = 0.0
    return r    

def reject_reward(completion: str, solution: list[str]) -> list[float]:
    should_reject = solution[0] == "<reject>"
    is_reject = rejection_patterns.search(completion)
    if (should_reject and is_reject) or (not should_reject and not is_reject):
        reward = 1.0
    else:
        reward = 0.0
    return reward

def behave_reward(completion: str, solution: list[str], enable_target_reject=True) -> list[float]:
    should_reject = solution[0] == "<reject>"
    if should_reject:
        correct_answer = solution[2]
    else:
        correct_answer = solution[1]
    rouge_l = compute_rouge_l(completion, correct_answer)
    answer_correct = rouge_l > 0.3
    if answer_correct:
        if (should_reject and enable_target_reject) or not should_reject:
            reward = 1.0
        else:
            reward = 0.0
    else:
        reward = 0.0
    return reward

def reject_behave_reward(completion, solution, reject_ratio=0.7, enable_target_reject=True):
    # Stage 1: Rejection stage
    reject_reward_value = reject_reward(completion, solution)
    
    # Stage 2: Behavior stage
    behave_reward_value = behave_reward(completion, solution, enable_target_reject)
    
    # Combine rewards
    overall_reward = reject_ratio * reject_reward_value + (1 - reject_ratio) * behave_reward_value
    
    return {
        "overall": overall_reward,
        "reject_reward": reject_reward_value,
        "behave_reward": behave_reward_value   
    }