# Project README

## Overview
This project supports model training and inference using datasets such as **ToxiBenchCN**, **Bitabuse**, and **Bitviper**.  
The training and inference codes are located under the `src/` directory.

---

## Data Preparation

### Datasets
- **ToxiBenchCN**: [https://github.com/thomasyyyoung/ToxiBenchCN](https://github.com/thomasyyyoung/ToxiBenchCN)  
- **Bitabuse / Bitviper**: [https://github.com/CAU-AutoML/Bitabuse](https://github.com/CAU-AutoML/Bitabuse)  
- Original data should be downloaded from the above repositories.

### Preprocessing / Modifications
- Code for preprocessing and data modifications can be found in `src/data`.

---

## Training

### Configuration
Before training, create an experiment configuration file `config/your_exp.yaml`.  
Example:

```yaml
seed: 44
cuda_visible_devices: "3"

# Dataset & Model
dataset_name: jwengr/ToxiBenchCN
pretrained_model_path: ""
base_model_name: Qwen/Qwen3-0.6B-Base
n_tokens_per_char: 4
input_chars: ""
target_chars: ""

# Training Hyperparameters
mini_batch_size: 32
n_batch: 1
epochs: 10
learning_rate: 0.0001
use_bntd: true
sliding_window: 12
neftune_alpha: 0
use_qlora: false
lora_r: 16
lora_alpha: 32

# Sequence Lengths
train_max_length: 128
valid_max_length: 128
inference_sentence_max_length: 64
inference_sentence_min_length: 32
inference_sentence_n_overlap: 3

prefix: ""
```


## Training Command
python train_bind_single_gpu.py config/your_exp.yaml

Inference Command
python inference_bind.py --config=config/your_exp.yaml --checkpoint=checkpoints/your_checkpoint.ckpt

Directory Structure Example
├── config/                # Experiment YAML files
├── checkpoints/           # Trained model checkpoints
├── src/
│   ├── data/              # Data preprocessing code
│   ├── train_bind_single_gpu.py
│   └── inference_bind.py
└── README.md
