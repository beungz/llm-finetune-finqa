import torch
import re
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from Scripts.make_dataset import parse_ground_truth



def extract_answer(text):
    """Extract the first numeric value following the 'FINAL ANSWER:' tag in the generated text."""

    # Split the text at the first occurrence of the target tag
    parts = re.split(r"FINAL ANSWER:\s*", text, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        # Isolate the section immediately after the tag
        section_text = parts[1]
        
        # Find all numbers or percentages in this section
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*%?", section_text)
        
        # Return the first number found to prevent grabbing hallucinated extra math
        if numbers:
            return numbers[0].replace(",", "")

    # Fallback: If the tag is missing, return the very last number found in the entire text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*%?", text)
    return numbers[-1].replace(",", "") if numbers else None



def ask_model(sample, tokenizer, model, max_new_tokens=200):
    """Format a FinQA sample, prompt the model, and extract the numeric prediction."""

    # Construct the context paragraph and format the table into a readable string
    context = " ".join(sample["pre_text"]) + " " + " ".join(sample["post_text"])
    table_str = ""
    for row in sample["table"]:
        if isinstance(row, list):
            table_str += " | ".join([str(x) for x in row]) + "\n"
        else:
            table_str += str(row) + "\n"

    question = sample["qa"]["question"]

    # Define the strict few-shot examples to enforce the 'FINAL ANSWER' formatting
    few_shot_prompt = """
Example 1:
Question: What is the gross profit margin?
CALCULATION: 200 / 500 = 0.4.
FINAL ANSWER: 40.0%

Example 2:
Question: What was the percentage growth in assets?
CALCULATION: (1240 - 1000) / 1000 = 240 / 1000 = 0.24.
FINAL ANSWER: 24.0%

Example 3:
Question: What is the ratio of debt to equity?
CALCULATION: 150 / 400 = 0.375.
FINAL ANSWER: 37.5%
---
    """

    # Build the chat template messages
    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst. Perform precise calculations. Always output the result as a single number or percentage in the FINAL ANSWER section."
        },
        {
            "role": "user",
            "content": f"{few_shot_prompt}\nNow analyze this:\nContext: {context}\nTable: {table_str}\nQuestion: {question}\n\nCALCULATION:\nFINAL ANSWER:"
        }
    ]

    # Tokenize the prompt and send it to the GPU
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the model's response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Calculate the exact length of the input prompt
    input_length = inputs["input_ids"].shape[1]
    
    # Slice the tensor to grab ONLY the newly generated tokens, ignoring the prompt
    generated_tokens = outputs[0][input_length:]
    
    # Decode the isolated generated tokens into readable text
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extract the final numeric answer from the response
    pred = extract_answer(response)

    return prompt, response, pred



def evaluate_model(dataset, tokenizer, model, n=20):
    """Evaluate the model's accuracy on a given dataset"""

    correct = 0
    total = 0
    tolerance = 1e-2

    # Iterate through the specified number of samples n
    for i in range(n):

        # Generate the prediction from the model
        prompt, response, pred = ask_model(dataset[i], tokenizer, model)

        # Parse the actual ground truth
        true = parse_ground_truth(dataset[i]["qa"]["answer"])

        # Skip evaluation if the ground truth is invalid or non-numeric
        if true is None:
            print(i, "skipped (non-numeric answer)")
            continue

        if pred is not None:
            # Parse both the predicted text and ground truth text into floats
            pred_num = parse_numeric_answer(pred)
            true_num = parse_numeric_answer(true)

            # Check if both are valid numbers
            if pred_num is not None and true_num is not None:
                # Mark as correct if it matches exactly (within tolerance) or within a 1% relative error margin
                if abs(pred_num - true_num) < tolerance or abs(pred_num - true_num)/max(abs(true_num), 1e-6) < 0.01:
                    correct += 1

        total += 1

        # Print ongoing progress and current accuracy
        print(i, pred, true)
        print(f"Current Accuracy: {correct / total:.2%}")

    # Print the final total accuracy
    if total > 0:
        print(f"\nFinal Accuracy: {correct / total:.2%}")



def parse_numeric_answer(value):
    """Parse a predicted string value into a numeric float, adjusting for percentages."""

    # Return None if value is null
    if value is None:
        return None

    # Return immediately if value is already a number
    if isinstance(value, (int, float)):
        return float(value)

    # Clean the string
    value = value.strip()

    # Detect and remove percentage signs
    is_percent = value.endswith("%")
    value = value.replace("%", "")

    # Attempt to cast to float and adjust mathematically if it was a percentage
    try:
        num = float(value)

        if is_percent:
            num = num / 100

        return num

    except:
        return None
    


def clear_vram():
    """Free up GPU VRAM by deleting ML objects and clearing CUDA caches."""

    objects = [
        "model",
        "tokenizer",
        "trainer",
        "train_dataset",
        "val_dataset",
        "data_collator",
        "optimizer",
        "lr_scheduler"
    ]

    # Delete global variables if they exist
    for obj in objects:
        if obj in globals():
            try:
                del globals()[obj]
            except Exception as e:
                pass

    # Reset PyTorch peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Python garbage collection
    gc.collect()

    # Empty PyTorch CUDA cache and IPC
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()



def load_model_for_evaluation(load_peft = False):
    """Load the Mistral base model in 4-bit quantization, and optionally attach the LoRA adapter."""

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Initialize the tokenizer and map the padding token to the EOS token
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Define 4-bit quantization settings to save VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load the base model onto the GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Attach the fine-tuned LoRA weights if load_peft is True
    if load_peft:
        model = PeftModel.from_pretrained(model, "./FinQA-LoRA-Adaptor")

    return tokenizer, model