import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Load model for fine-tuning
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    context = example['context']  # The financial table or context data
    qa_pairs = []

    # Loop through each question-answer pair
    for i, (question, answer) in enumerate(zip(example['questions'], example['answers'])):
        if i == 0:
            # Initial prompt with context and first question
            prompt = f"Context: {context}\nHistory:\nQ: {question}\nA: {answer}\n"
        elif i == len(example["questions"]) - 1:
            # Last prompt add final question and answer markings
            prompt = f"{qa_pairs[-1]['input_text']}\nQuestion to answer: {question}\nAnswer: {answer}"
        else:
            # For follow-up questions, use previous Q&A pairs to provide conversation history
            prompt = f"{qa_pairs[-1]['input_text']}Q: {question}\nA: {answer}\n"

        qa_pairs.append({
            "input_text": prompt + tokenizer.eos_token
        })

    return qa_pairs[-1]

# Load the ConvFinQA dataset
dataset = load_dataset("convfinqa.py")

# Apply formatting to the dataset
train_data = dataset["train"].map(
    format_example,
    remove_columns=["context", "questions", "answers"]
)
valid_data = dataset["validation"].map(
    format_example,
    remove_columns=["context", "questions", "answers"]
)

def tokenize_function(example):
    return tokenizer(
        example['input_text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize data
tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_valid_data = valid_data.map(tokenize_function, batched=True)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-finetuned-convfinqa",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Keep batch size low
    per_device_eval_batch_size=1,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,  # Use FP16 for lower memory usage
    gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
    max_grad_norm=1.0,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_valid_data,
    data_collator=data_collator,
)

# Train
if os.path.exists("./llama-finetuned-convfinqa"):
    train_result = trainer.train(resume_from_checkpoint=True)
else:
    train_result = trainer.train()

# compute and save train metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# compute and save evaluation metrics
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

model.save_pretrained("llama-finetuned-convfinqa")
tokenizer.save_pretrained("llama-finetuned-convfinqa")