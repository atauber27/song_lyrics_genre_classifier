import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from accelerate import Accelerator
import os

def main():
    torch.cuda.empty_cache()
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Devices:", torch.cuda.device_count())
    
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    accelerator = Accelerator()
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model, tokenizer = accelerator.prepare(model, tokenizer)
    train_file_path = "processed_train_dataset.tsv"
    eval_file_path = "processed_val_dataset.tsv" 

    train_df = pd.read_csv(train_file_path, sep="\t") 
    eval_df = pd.read_csv(eval_file_path, sep="\t")  

    # Cut down the training dataset to 10%
    #train_df = train_df.sample(frac=0.1, random_state=42)  # Adjust `frac` for the proportion you want

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=1024) 
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch", 
        num_train_epochs=4,
        fp16=True,  # Enable mixed precision
        logging_dir="./logs",
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=2,  # Simulates a larger batch size
        dataloader_num_workers=4,    
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )
    trainer.train()
    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

if __name__ == "__main__":
    main()
