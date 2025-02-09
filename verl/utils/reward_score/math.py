# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import numpy as np
import random
from .utils import extract_answer_math


def cosfn(t, T, eta_min, eta_max):
    """
    Implements the cosine function for reward scaling
    
    Args:
        t: Current value (e.g., generation length)
        T: Maximum value (e.g., maximum length)
        eta_min: Minimum value for scaling
        eta_max: Maximum value for scaling
    
    Returns:
        float: Scaled value based on cosine function
    """
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(t * np.pi / T))


def linear_length_reward_function(C, L_gen, L_max):
    if C:
        return  1 - 0.5 * L_gen / L_max
    else:
        return -1 + 0.5 * L_gen / L_max


def cosine_reward_function(C, L_gen, L_max):
    """
    Implements the reward function R(C, L_gen) based on Qwen2.5-Math-7B parameters
    
    Args:
        C: Correctness (True or False)
        L_gen: Generation length
        L_max: Maximum length
    
    Returns:
        float: Calculated reward
    """
    # Define hyperparameters
    r0_c = 2.0    # Reward for correct at L_gen = 0
    rL_c = 1.0    # Reward for correct at L_gen = L_max
    r0_w = -10.0  # Reward for wrong at L_gen = 0
    rL_w = 0.0    # Reward for wrong at L_gen = L_max
    r_e = -10.0   # Exceed length penalty

    # Handle the case where L_gen >= L_max first
    if L_gen >= L_max:
        return r_e
    
    # Handle correct case
    if C:
        return cosfn(L_gen, L_max, r0_c, rL_c)
    # Handle wrong case
    elif not C:
        return cosfn(L_gen, L_max, r0_w, rL_w)


def compute_score(solution_str, ground_truth, reward_type="classic", tokenizer=None, max_length=None) -> float:
    correct = False
    answer = "None"
    do_print = random.randint(1, 256) == 1
    try:
        answer = extract_answer_math(solution_str)
        if answer == ground_truth['target']:
            correct = True
    except Exception as e:
        print(e)

    if reward_type == "cosine":
        length = min(len(tokenizer.tokenize(solution_str)), max_length)  # Clip to max_length
        retval = cosine_reward_function(correct, length, max_length)
    elif reward_type == "classic_length_penalty":
        length = min(len(tokenizer.tokenize(solution_str)), max_length)  # Clip to max_length
        retval = linear_length_reward_function(correct, length, max_length)
    elif reward_type == "classic":
        retval = 1. if correct else -1
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    if do_print:
        print(f"--------------------------------")
        print(f"Question: {ground_truth['question']}")
        print(f"Solution: {solution_str}")
        print(f"Ground Truth: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"{'Correct' if correct else 'Incorrect'} answer! Reward = {retval}")

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
