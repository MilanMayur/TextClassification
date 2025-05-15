from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from huggingface_hub import upload_file
import evaluate
import torch
import os

# Load dataset
dataset = load_dataset("sayakpaul/genre-classification-imdb")

# Filter rows BEFORE tokenization
def is_valid(example):
    parts = example["text"].split(" ::: ")
    return len(parts) >= 4 and parts[2].strip().lower() in ["comedy", "drama", "horror", "action"]

dataset["train"] = dataset["train"].filter(is_valid)
dataset["test"] = dataset["test"].filter(is_valid)

# Tokenizer and labels
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
genres = ['comedy', 'drama', 'horror', 'action']
label2id = {g: i for i, g in enumerate(genres)}
id2label = {i: g for g, i in label2id.items()}

# Batched tokenization
def tokenize_batch(batch):
    input_texts = []
    labels = []

    for text in batch["text"]:
        parts = text.split(" ::: ")
        genre = parts[2].strip().lower()
        plot = parts[3].strip()

        input_texts.append(plot)
        labels.append(label2id[genre])

    tokenized = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = labels
    return tokenized

# Tokenize dataset
tokenized_dataset = dataset.map(
                                tokenize_batch,
                                batched=True,
                                remove_columns=dataset["train"].column_names
                            )

# Format for PyTorch
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
                                    "bert-base-uncased", 
                                    num_labels=len(genres), 
                                    id2label=id2label,
                                    label2id=label2id
                                ) 

# Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Training arguments
training_args = TrainingArguments(
                    output_dir='./results',         
                    eval_strategy="epoch",            
                    save_strategy="epoch",
                    logging_dir='./logs',          
                    learning_rate=2e-5,
                    num_train_epochs=3,             
                    per_device_train_batch_size=8,    
                    per_device_eval_batch_size=16,    
                    weight_decay=0.01,
                    load_best_model_at_end=True,
                    logging_steps=10,              
                    push_to_hub=True,
                    hub_model_id="milanmayur20/imdb-genre-bert", 
                    hub_strategy="end",
                    disable_tqdm=False,
                    logging_first_step=True
                )

print("TrainingArguments set correctly!")

# Trainer
trainer = Trainer(
                    model=model,                          
                    args=training_args,                        
                    train_dataset=tokenized_dataset["train"],    
                    eval_dataset=tokenized_dataset["test"],       
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=compute_metrics
                )

# Train model
trainer.train()

# Save
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Upload to Hugging Face Hub
trainer.push_to_hub()
