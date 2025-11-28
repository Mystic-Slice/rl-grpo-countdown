import pandas as pd
from datasets import load_dataset

def process_example(sample):

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': "Answer the following question. Provide the reasoning between <reasoning> and </reasoning> symbols. Provide the final answer (an expression) between <answer> and </answer> symbols.\n"\
                f"Using the numbers {sample['nums']}, create an equation that equals {sample['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once." \
        }
    ]

    return {
        'prompt': PROMPT_MESSAGES,
    }

# gsm8k
# def process_example(sample):
#     answer_extracted = int(sample['answer'].split('####')[1].strip().replace(',', ''))

#     PROMPT_MESSAGES = [
#         {
#             'role': 'user',
#             'content': "Answer the following question. Provide the reasoning between <reasoning> and </reasoning> symbols. Provide the final answer (an integer) between <answer> and </answer> symbols.\n"\
#                 f"Question: {sample['question']}\n" \
#         }
#     ]

#     return {
#         'prompt': PROMPT_MESSAGES,
#         'answer_extracted': answer_extracted
#     }

# mmlu
# def process_example(sample):
#     choices = sample['choices']
#     options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
#     options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
#     correct_option = chr(65 + sample['answer'])

#     PROMPT_MESSAGES = [
#         {
#             'role': 'user',
#             'content': "Answer the following question. Provide the reasoning between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."\
#                 f"Question: {sample['question']}\n" \
#                 f"Options: \n{options_str}"
#         }
#     ]

#     return {
#         'prompt': PROMPT_MESSAGES,
#         'correct_option': correct_option
#     }

def get_data():
    ds = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4')['train']
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    ds = ds.train_test_split(test_size=0.001, seed=42)

    return ds