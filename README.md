# Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization

This repository contains PyTorch source code for EMNLP 2024 paper [Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization](https://arxiv.org/abs/2406.15524).

Our implementation is based on [EBFT](https://github.com/sunggo/EBFT/tree/main), [Wanda](https://github.com/locuslab/wanda), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), and [LLM-QAT](https://github.com/facebookresearch/LLM-QAT).

## Environments

### Python
- python 3.9

### Dependencies
```bash
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage

### Basic usage (for LLaMA)
LR
```bash
python main.py --config=./configs/llama.py
```

LR + GP
```bash
python main.py --config=./configs/llama.py --config.use_gp=True
```

LR + GP + CR
```bash
python main.py --config=./configs/llama.py --config.use_gp=True --config.use_cr=True
```

### OPT model

```bash
python main.py --config=./configs/opt.py
```

### Self-generated data
First, generate the data as follows.
```bash
python generate_data.py --config=./configs/data.py
```

Then, set config.self_nsamples to be a positive number.
```bash
python main.py --config=./configs/llama.py --config.self_nsamples=256
```

### Zero-shot performance evaluation

First, download the directory from the [link](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view) provided from the Wanda repository.
Next, change the directory name to lm_eval.
Then, set config.eval_zero_shot as True.
```bash
python main.py --config=./configs/llama.py --config.eval_zero_shot=True
```