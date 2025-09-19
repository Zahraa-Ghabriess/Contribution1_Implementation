# Install dependencies (uncomment if running for the first time)
# !pip install transformers torch datasets

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch


# 1. Load dataset (IMDb sentiment as an example)
dataset = load_dataset("imdb")

# 2. Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))  # use subset
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 4. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    #do_eval=True,  # enables evaluation
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=test_dataset,
#)

# 5. Train model
#trainer.train()

# 6. Evaluate model
'''eval_result = trainer.evaluate()
print("Evaluation:", eval_result)

# 7. Test prediction
sample_text = "This movie was absolutely fantastic!"
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

print("Predicted label:", "Positive" if prediction == 1 else "Negative")'''
