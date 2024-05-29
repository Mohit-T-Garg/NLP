import sys
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import nltk
from transformers import DataCollatorForSeq2Seq
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType


nltk.download('punkt')  # Download the punkt tokenizer if not already downloaded
metric = evaluate.load("rouge")

prefix = "Summarize this article for a layman.\n"

def preprocess_function(example):
    sections = example["article"].split('\n')
    abstract = sections[0]
    conclusion = sections[-1]       
    
    keywords = ", ".join(example["keywords"])
    concatenated_text = f"{prefix} Keywords: {keywords} \n Abstract: {abstract} \n Conclusion: {conclusion}"

    model_inputs = tokenizer(concatenated_text, max_length=512, truncation=True)
    
    labels = tokenizer(text_target=example["lay_summary"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <path_to_data> <path_to_save>")
        sys.exit(1)

    path_to_data = sys.argv[1]
    path_to_save = sys.argv[2]
    
    rouge = evaluate.load('rouge')

    # Load the dataset
    eLife_train = load_dataset("json", data_files= path_to_data + "/eLife_train.jsonl")
    eLife_val = load_dataset("json", data_files= path_to_data + "/eLife_val.jsonl")

    PLOS_train = load_dataset("json", data_files= path_to_data + "/PLOS_train.jsonl")
    PLOS_val = load_dataset("json", data_files= path_to_data + "/PLOS_val.jsonl")


    model_name="google/flan-t5-small"
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Map the preprocessing function across our dataset
    tokenized_eLife_train = eLife_train.map(preprocess_function)
    tokenized_eLife_val = eLife_val.map(preprocess_function)

    # Map the preprocessing function across our dataset
    tokenized_PLOS_train = PLOS_train.map(preprocess_function)
    tokenized_PLOS_val = PLOS_val.map(preprocess_function)

    # Concatenate the tokenized datasets for training
    train_dataset = concatenate_datasets([tokenized_eLife_train['train'], tokenized_PLOS_train['train']])

    # Concatenate the tokenized datasets for validation
    val_dataset = concatenate_datasets([tokenized_eLife_val['train'], tokenized_PLOS_val['train']])

    # Shuffle the merged training dataset
    train_dataset = train_dataset.shuffle(seed=42)

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )
    peft_model = get_peft_model(original_model, 
                                lora_config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model)

    peft_training_args = TrainingArguments(
        output_dir=path_to_save,
        auto_find_batch_size=True,
        learning_rate=1e-4, # higher learning rate
        num_train_epochs=3,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True
    )


        
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    peft_trainer.train()


    peft_trainer.model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)
