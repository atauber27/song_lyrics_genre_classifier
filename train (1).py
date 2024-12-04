from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

train_file_path = "/home/eheakins/NLPScholar/songGeneration/data/processed_train_dataset.tsv"
eval_file_path = "/home/eheakins/NLPScholar/songGeneration/data/processed_val_dataset.tsv" 

train_df = pd.read_csv(train_file_path, sep="\t") 
eval_df = pd.read_csv(eval_file_path, sep="\t")  

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the trained model
model.save_pretrained("./fine_tuned_llama_1epoch_gpu")
tokenizer.save_pretrained("./fine_tuned_llama_1epoch_gpu")
