# AIPI 590: Project 1: Enhancing LLM's Mathematical Reasoning for Financial Questions
### **Author**: Matana Pornluanprasert

This project focus on fine-tuning of 7-billion parameter language model, Mistral-7B-Instruct-v0.3, to perform precise mathematical reasoning on complex financial questions.<br>
<br>


***
# The Property of Interest<br>
LLMs are great at summarizing text but often hallucinate when calculating numbers. This project aims to fine-tune the model with focus on financial numerical reasoning and see if a smaller LLM model could be trained to answer to financial questions correctly.<br>
<br>


***
# Data Source<br>
The training, validation, and test data are all from FinQA, a highly regarded financial dataset containing thousands of expert-annotated questions built against real S&P 500 earnings reports.<br>
https://finqasite.github.io/<br>
After cleaning and dataset split, the dataset size is training : validation : test = 5,938 : 547 : 547<br>
<br>


***
# Model Selection<br>
Mistral-7B-Instruct-v0.3 is chosen for the best balance of reasoning capability relative to its size. Instead of Base Model, Instruct Model is used as it was fine-tuned to follow commands and output specific structures. With 7 billion parameters, the model can be fine-tuned on RTX5060 with 16GB VRAM.<br>
<br>


***
# Fine-tuning Approach<br>

| Optimization Technique | Setting Used | Purpose |
| :--- | :--- | :--- |
| Quantization | 4-bit (NF4) | Squeezed the 7B parameter model into 16GB of VRAM. |
| Adapter (QLoRA) | r=16, alpha=32 | Targeted attention and projection layers for efficient learning. |
| Context Window | 1,750 Tokens | Captured 95% of long financial documents without truncating the final answers. |
| Batch Strategy | Batch=2, Grad Acc=4 | Maintained an effective batch size of 8 to prevent out-of-memory (OOM) crashes. |

Due to large file size (160 MB), LoRA adapter file, adapter_model.safetensors, are excluded from this github repos. It can be downloaded separately from the following Huggingface repos:<br>
https://huggingface.co/beung/aipi590-project1/resolve/main/adapter_model.safetensors
<br>
After download complete, please put it in FinQA-LoRA-Adaptor folder.<br>
<br>


***
# Results and Conclusions<br>

| Model | Accuracy Score |
| :--- | :--- |
| Pre-Training Baseline | 32.54% |
| Post-Training (FinQA LoRA) | 64.90% |

<br>
We successfully doubled the accuracy of the base model, taking it from 32.54% to 64.90%. With precise prompt formatting, and memory-efficient LoRA training, a 7B parameter model can be transformed into a financial question solver.<br>
<br>


***
# Ethics statement<br>
This project is intended for research and educational purposes in large language model. All data collection, model training, and deployment are conducted with respect for privacy and copyright. Care has been taken to avoid misuse of the model and to ensure responsible use of the technology, particularly in relation to surveilance, personal data, and public safety.<br>
<br>


***
# Requirements and How to run the code

### **Requirements**:<br>
```
# torch==2.10.0+cu128
transformers==4.57.6
datasets==4.7.0
peft==0.18.1
bitsandbytes==0.49.2
accelerate==1.13.0
```
<br>

### **How to run the code**:<br>
***

To finetune and evaluate the model: type the followings in the terminal<br>

On Windows:<br>

```
py main.py
```

On other systems:<br>

```
python main.py
```

<br>


***
