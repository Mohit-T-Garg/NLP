import sys
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
from peft import PeftModel, PeftConfig
import evaluate
import pandas as pd
import numpy as np
import nltk
from transformers import DataCollatorForSeq2Seq
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

nltk.download('punkt')  # Download the punkt tokenizer if not already downloaded
metric = evaluate.load("rouge")

prefix = "Summarize this article for a layman.\n"

def preprocess_eval(example):
    sections = example["article"].split('\n')
    abstract = sections[0]
    conclusion = sections[-1]       
    
    keywords = ", ".join(example["keywords"])
    concatenated_text = f"{prefix} Keywords: {keywords} \n Abstract: {abstract} \n Conclusion: {conclusion}"

    model_inputs = tokenizer(concatenated_text, max_length=512, truncation=True)
    return model_inputs



# Define a function to generate summaries from input IDs
def generate_summary(input_ids, model, tokenizer):
    max_output_length = 512  # Set your desired maximum length here
    output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=512))
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def process_input_ids(input_ids_list, model, tokenizer):
    summaries = []
    for input_ids in tqdm(input_ids_list):
        summary = generate_summary(input_ids, model, tokenizer)
        summaries.append(summary)
    return summaries



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train.py <path_to_data> <path_to_save>")
        sys.exit(1)

    path_to_data = sys.argv[1]
    path_to_model = sys.argv[2]
    path_to_save = sys.argv[3]


    # Load the dataset
    eLife_test = load_dataset("json", data_files= path_to_data + "/eLife_test.jsonl")

    PLOS_test = load_dataset("json", data_files= path_to_data + "/PLOS_test.jsonl")


    model_name="google/flan-t5-small"
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                        path_to_model, 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)

    tokenized_eLife_test = eLife_test.map(preprocess_eval)

    # Map the preprocessing function across our dataset
    tokenized_PLOS_test = PLOS_test.map(preprocess_eval)

    eLife_test_data = tokenized_eLife_test['train']
    PLOS_test_data = tokenized_PLOS_test['train']

    input_ids_list = [torch.tensor(input_ids).view(1, -1) for input_ids in eLife_test_data["input_ids"]]
    eLife_summaries = process_input_ids(input_ids_list, peft_model, tokenizer)

    output_file = path_to_save + "/elife.txt"
    with open(output_file, "w") as f:
        for summary in eLife_summaries:
            f.write(summary + "\n")

    input_list = [torch.tensor(input_ids).view(1, -1) for input_ids in PLOS_test_data["input_ids"]]

    PLOS_summaries = process_input_ids(input_list, peft_model, tokenizer)

    output_file = path_to_save + "/plos.txt"
    with open(output_file, "w") as f:
        for summary in eLife_summaries:
            f.write(summary + "\n")



    

