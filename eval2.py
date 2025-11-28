from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_data
import datasets
from tqdm import tqdm
import re
import time
import glob
import os 

def extract_answer(completion):
    match = re.search(r"<answer>(.*)</answer>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'

FOLDERS = [
    'rl_equation_first', 
    'rl_equation_base_config',
    'rl_equation_base_config_alllinear',
    'rl_equation_base_config_alllinear_kl_largebeta',
    'rl_equation_base_config_kl',
    'rl_equation_base_config_kl_high_lr', 
    'rl_equation_base_config_kl_largebeta',
    'rl_equation_think', 
    'rl_equation_think_base_config_backup',
    'base_config_kl_0.1_beta',
]

for folder in FOLDERS:
    checkpoints = glob.glob(f"{folder}/checkpoint-*")
    
    for checkpoint in checkpoints:
        print(f"Now running {checkpoint}")
        
        checkpoint_number = int(checkpoint.split('-')[-1])
        checkpoint_output = f"eval_outputs_left_padding/{folder}/checkpoint-{checkpoint_number}"
        if os.path.exists(checkpoint_output):
            print("Already exists")
            continue


        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        print("Base model loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            padding_side='left',
        )
        print("Tokenizer loaded")

        model = PeftModelForCausalLM.from_pretrained(
            model, 
            checkpoint
        )
        print("Peft model Loaded")

        # print(model)

        # Create pipeline with batching enabled
        pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
        )

        print(pipe.device)

        ds = get_data()

        print(ds['test'])

        test_ds = ds['test']

        test_ds = test_ds.shuffle(seed=42).select(range(100))

        # Prepare all prompts
        prompts = [sample['prompt'] for sample in test_ds]

        # Batch process with pipeline
        print("Generating responses...")
        start = time.time()
        responses = pipe(
                prompts,
                max_new_tokens=1024,
                batch_size=8,  # Adjust based on your GPU memory
                return_full_text=False  # Only return generated text, not input
            )
        print("Took", time.time() - start, 'secs')

        # Process results
        final_records = []
        for sample, response in tqdm(zip(test_ds, responses), total=len(test_ds)):
            generated_text = response[0]['generated_text']
            answer = extract_answer(generated_text)
            
            # print("Prompt: ", sample['prompt'])
            # print("Model response: ", generated_text)
            # print("Model Answer Extracted: ", answer)
            # print("Correct Answer:", sample['correct_option'])
            # print("="*100)
            
            record = dict(sample)
            record['model_response'] = generated_text
            record['model_answer'] = answer
            final_records.append(record)

        out_ds = datasets.Dataset.from_list(final_records)

        out_ds.save_to_disk(checkpoint_output)
