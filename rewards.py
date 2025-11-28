import re

def extract_answer(completion):
    match = re.search(r"<answer>(.*)</answer>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards_list = [1.0 if match else -1.0 for match in matches]
    return rewards_list

def accuracy_reward(completions, nums, target, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for completion, nums_i, target_i in zip(completion_contents, nums, target):
        try:
            # Check if the format is correct
            equation = extract_answer(completion)

            if equation is None:
                rewards.append(-1)
                continue

            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(nums_i):
                rewards.append(-1)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(-1)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(target_i)) < 1e-5:
                rewards.append(1)
            else:
                rewards.append(-1)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(-1)
    format_rewards = format_reward(completions)
    for comp, reward, form_reward in zip(completion_contents, rewards, format_rewards):
        print(comp)
        print("Accuracy Reward: ", reward)
        print("Format Reward: ", form_reward)
        print("="*50)
    print("Accuracy Rewards: ", rewards)
    print("Format Rewards: ", format_rewards)
    return rewards

# gsm8k
# def accuracy_reward(completions, answer_extracted, **kwargs):
#     completion_contents = [completion[0]["content"] for completion in completions]
#     answers = [extract_answer(comp) for comp in completion_contents]

#     rewards = []
#     for ans, correct_ans in zip(answers, answer_extracted):
#         try:
#             ans_num = int(ans)
#             if ans_num == correct_ans:
#                 rewards.append(1)
#             else:
#                 rewards.append(-1)
#         except:
#             rewards.append(-1)

#     for comp, reward in zip(completion_contents, rewards):
#         print(comp)
#         print("Reward: ", reward)
#         print("="*50)
#     return rewards

# mmlu
# def accuracy_reward(completions, correct_option, **kwargs):
#     completion_contents = [completion[0]["content"] for completion in completions]
#     answers = [extract_answer(comp) for comp in completion_contents]

#     rewards = [1 if ans == correct_ans else -1 for (ans, correct_ans) in zip(answers, correct_option)]

#     for comp, reward in zip(completion_contents, rewards):
#         print(comp)
#         print("Reward: ", reward)
#         print("="*50)
#     return rewards
