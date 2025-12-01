# rl-grpo-countdown

**Learning GRPO: Experiments and Insights from Fine-Tuning an LLM**

This repository contains code, configurations, and results for experiments described in the blog post *Learning GRPO: Experiments and Insights from Fine-Tuning an LLM*. The goal is to explore training a language model using Group Relative Policy Optimization (GRPO) / RL with Verifiable Rewards (RLVR) on a simple toy task called the [Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).  

Link to Blog Post: https://mystic-slice.github.io/blogs/rl_grpo/

## Repository Contents

- `train.py` - The main training script
- `data.py` - load and process samples for training and eval
- `model.py` - load model and create LoRA adapters
- `rewards.py` - reward functions used during training
- `eval_checkpoints.py` - evaluate trained models 
- `analyse.ipynb` - analyze model evaluations

## How to run / reproduce

1. Install dependencies 
```py
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
2. Train
```py
python train.py
```
3. Eval
```py
# Add the output directory to the FOLDERS list in line 19 in eval_checkpoints.py
python eval_checkpoints.py
```
See the aggregated results in `analyse.ipynb`