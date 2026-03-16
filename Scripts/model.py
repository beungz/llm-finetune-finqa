import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset



def format_steps(program):
    """Parse and execute a sequence of mathematical operations to generate readable calculation steps and the final result."""
    
    # Extract operations using regex
    if isinstance(program, str):
        steps = re.findall(r'[a-zA-Z_]+\(.*?\)', program)
    else:
        steps = program

    results = []
    calculations = []

    def resolve(x):
        """Helper function to resolve string arguments into numeric values or previous step results."""
        x = x.strip()

        # Check if x references a previous calculation step
        if x.startswith("#"):
            try:
                idx = int(x[1:])
                return results[idx]
            except (IndexError, ValueError):
                return 0.0
            
        # Check if x is a constant
        if x.startswith("const_"):
            try:
                return float(x.replace("const_", ""))
            except ValueError:
                return 0.0
        
        # Parse standard numbers, remove currency symbols and commas
        try:
            return float(x.replace("$", "").replace(",", ""))
        except ValueError:
            return 0.0

    # Loop through each step to perform the math
    for step in steps:
        if "(" not in step or ")" not in step:
            continue
        
        # Split the operation type and its arguments
        op = step.split("(")[0].strip().lower()
        args_raw = step.split("(")[1].replace(")", "").split(",")
        args = [resolve(a) for a in args_raw if a.strip()]

        # Skip operations that do not have sufficient number of arguments
        if op in ["add", "subtract", "multiply", "divide", "exp", "greater"] and len(args) < 2:
            continue

        try:
            # Do the math and format the output string
            if op == "add":
                val = args[0] + args[1]
                expr = f"{args[0]} + {args[1]}"
            elif op == "subtract":
                val = args[0] - args[1]
                expr = f"{args[0]} - {args[1]}"
            elif op == "multiply":
                val = args[0] * args[1]
                expr = f"{args[0]} * {args[1]}"
            elif op == "divide":
                val = args[0] / args[1] if args[1] != 0 else 0.0
                expr = f"{args[0]} / {args[1]}"
            elif op == "exp":
                val = args[0] ** args[1]
                expr = f"{args[0]} ^ {args[1]}"
            elif op == "greater":
                val = max(args[0], args[1])
                expr = f"max({args[0]}, {args[1]})"
            elif op == "table_average":
                val = sum(args) / len(args) if len(args) > 0 else 0.0
                expr = f"average({', '.join(map(str, args))})"
            elif op == "table_sum":
                val = sum(args)
                expr = f"sum({', '.join(map(str, args))})"
            elif op in ["table_max", "table_greater"]:
                val = max(args) if args else 0.0
                expr = f"max({', '.join(map(str, args))})"
            elif op == "table_min":
                val = min(args) if args else 0.0
                expr = f"min({', '.join(map(str, args))})"
            else:
                continue
            
            # Store the resulting value and the expression
            results.append(val)
            calculations.append(f"{expr} = {val}")
            
        except Exception:
            continue
    
    if not results:
        return "No valid calculation steps found.", 0.0
        
    return "\n".join(calculations), results[-1]



def build_prompt(tokenizer, sample):
    """Build a chat prompt for the model"""

    # Concatenate the pre-text and post-text
    context = " ".join(sample["pre_text"]) + " " + " ".join(sample["post_text"])
    table_str = ""

    # Format the table rows
    for row in sample["table"]:
        if isinstance(row, list):
            table_str += " | ".join([str(x) for x in row]) + "\n"
        else:
            table_str += str(row) + "\n"

    # Get the question and math calculation steps
    question = sample["qa"]["question"]
    program = sample["qa"]["program"]
    calc_steps, answer = format_steps(program)

    # Define the system prompt
    messages = [
        {
            "role": "system", 
            "content": "You are a financial analyst. Perform precise calculations. Always output the result as a single number or percentage in the FINAL ANSWER section."
        },
        {
            "role": "user", 
            "content": f"Context: {context}\nTable: {table_str}\nQuestion: {question}\n\nCALCULATION:\nFINAL ANSWER:"
        },
        {
            "role": "assistant",
            "content": f"CALCULATION:\n{calc_steps}\n\nFINAL ANSWER:\n{answer}"
        }
    ]

    # Apply Mistral Model's chat template (tokenize=False as the trainer will handle tokenization)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return full_text



def format_dataset(tokenizer, dataset):
    """Loop through a raw dataset and format each sample into a text prompt ready for tokenization."""
    prompts = []
    for sample in dataset:
        try:
            # Build the formatted conversation string
            prompt = build_prompt(tokenizer, sample)
            prompts.append({"text": prompt})
        except Exception as e:
            # Skip samples that fail the formatting process
            continue
    return prompts



def prepare_trainer(train_data, val_data):
    """Initialize the model, tokenizer, and trainer with 4-bit quantization and LoRA configurations."""

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Setup tokenizer and map the padding token to the EOS token
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_align(example):
        """Tokenize the text and mask padding tokens"""
        outputs = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=1750, 
            add_special_tokens=False
        )
        
        # Copy the input_ids to create the target labels
        labels = outputs["input_ids"].copy()
        
        # Mask the padding tokens with -100 using the attention_mask
        labels = [
            label if mask == 1 else -100 
            for label, mask in zip(labels, outputs["attention_mask"])
        ]
        
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": labels
        }

    # Format raw dictionaries into formatted text prompts
    train_prompts = format_dataset(tokenizer, train_data)
    val_prompts = format_dataset(tokenizer, val_data)

    # Convert text prompts into Dataset objects
    train_raw = Dataset.from_list(train_prompts)
    val_raw = Dataset.from_list(val_prompts)

    # Apply tokenization and remove the raw text column
    train_dataset = train_raw.map(tokenize_and_align, batched=False, remove_columns=["text"])
    val_dataset = val_raw.map(tokenize_and_align, batched=False, remove_columns=["text"])

    # Define 4-bit quantization config to save VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load the base model onto the GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Align padding tokens and disable caching to save memory during training
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Enable gradient checkpointing to prevent Out-Of-Memory errors
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Attach the LoRA adapter to the base model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments. Note that this is tuned for RTX5060 with 16GB VRAM
    training_args = TrainingArguments(
        output_dir="./FinQA-LoRA",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=20,
        save_steps=300,
        eval_strategy="steps",
        eval_steps=300,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=True,
        load_best_model_at_end=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    return tokenizer, model, trainer



def finetune_model(tokenizer, model, trainer):
    """Finetune the model and save the fine-tuned LoRA adapter and tokenizer"""
    trainer.train()
    model.save_pretrained("./FinQA-LoRA-Adaptor")
    tokenizer.save_pretrained("./FinQA-LoRA-Adaptor")