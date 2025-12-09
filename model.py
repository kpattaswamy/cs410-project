import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

reddit_data = pd.read_csv("preprocessed-dataset/labeled.csv")
reddit_data = reddit_data[reddit_data['next_day_movement'] != -1].copy()
reddit_data['label'] = reddit_data['next_day_movement'].map({0: 0, 1: 1})

reddit_data['text'] = reddit_data['title'].fillna('') + ' ' + reddit_data['selftext'].fillna('')
reddit_data = reddit_data[reddit_data['text'].str.strip() != ''].copy()

print(f"Dataset size after filtering: {len(reddit_data)}")

class StockDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    reddit_data['text'].values,
    reddit_data['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=reddit_data['label'].values
)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = StockDataset(train_texts, train_labels, tokenizer)
test_dataset = StockDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
).to(device)

# Training setup
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

print("Starting training...")
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/3, Average Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1: {f1:.4f}")

os.makedirs("./bert_stock_predictor_final", exist_ok=True)

# Save model
torch.save(model.state_dict(), "./bert_stock_predictor_final/pytorch_model.bin")
tokenizer.save_pretrained("./bert_stock_predictor_final")
