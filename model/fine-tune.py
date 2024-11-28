#%% Imports and Setup
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Configure Device and Paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_PATH = os.path.join(project_root, 'data', 'all_sentiments.csv')
MODEL_SAVE_PATH = os.path.join(project_root, 'model', 'nfl_sentiment_model_final')

print("Data path:", DATA_PATH)
print("Model save path:", MODEL_SAVE_PATH)

#%% Load and Prepare Data#%% Load and Prepare Data
try:
    # First attempt with automatic delimiter detection
    df = pd.read_csv(DATA_PATH, sep=None, engine='python', on_bad_lines='skip')
except Exception as e:
    print(f"First attempt failed: {e}")
    try:
        # Second attempt with explicit CSV handling
        df = pd.read_csv(
            DATA_PATH,
            encoding='utf-8',
            escapechar='\\',
            quoting=1,  # csv.QUOTE_ALL
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Second attempt failed: {e}")
        # Last resort: read with minimal processing
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', engine='python')
# Print data info
print("\nDataset Info:")
print(df.info())
print("\nColumns found:", df.columns.tolist())

# Ensure we have required columns
required_columns = ['text', 'sentiment']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Keep only necessary columns
df = df[required_columns]

# Data cleanup
df = df.dropna()  # Remove rows with missing values
df = df[df['sentiment'].isin(['negative', 'neutral', 'positive'])]  

# Convert sentiment labels to integers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(label_map)

print("\nSaving sample of processed data...")
df.head(10).to_csv('sample_processed_data.csv', index=False)

print("\nFinal dataset shape:", df.shape)

#%% Define Dataset Class
class NFLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

#%% Initialize Model and Tokenizer
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=99, stratify=df['label'])
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)
model.to(device)

#%% Create Datasets and Dataloaders
train_dataset = NFLDataset(train_df['text'].values, train_df['label'].values, tokenizer)
eval_dataset = NFLDataset(eval_df['text'].values, eval_df['label'].values, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

#%% Training Setup
num_epochs = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

#%% Training Functions
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        _, predictions = torch.max(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.shape[0]
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader), correct_predictions / total_predictions

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            _, predictions = torch.max(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.shape[0]
            total_loss += loss.item()
    
    return total_loss / len(dataloader), correct_predictions / total_predictions

#%% Training Loop
best_accuracy = 0
training_stats = []

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler)
    eval_loss, eval_acc = evaluate(model, eval_dataloader)
    
    print(f'Train Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}')
    print(f'Val Loss: {eval_loss:.4f} Accuracy: {eval_acc:.4f}')
    
    if eval_acc > best_accuracy:
        best_accuracy = eval_acc
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print("Saved new best model")
    
    training_stats.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'eval_loss': eval_loss,
        'eval_acc': eval_acc
    })

#%% Plot Results
stats_df = pd.DataFrame(training_stats)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(stats_df['train_loss'], label='train')
plt.plot(stats_df['eval_loss'], label='val')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(stats_df['train_acc'], label='train')
plt.plot(stats_df['eval_acc'], label='val')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(project_root, 'model', 'training_history.png'))
plt.show()

#%% Test Function
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return reverse_label_map[prediction.item()]

#%% Test the Model
test_texts = [
    "Love watching the Colts play! Great team!",
    "Terrible game today, very disappointed",
    "Just another regular game from the Colts",
    "Dak Prescott injuring his hamstring has positioned Dallas Cowboys into a tough spot with playoff hopes possibly over. #NFL"
]

print("\nTesting model predictions:")
for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Predicted sentiment: {sentiment}")

# %%
