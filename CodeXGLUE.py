import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os
from torch import tensor

def save_functions_to_files(dataset, text_files_dir):
    Path(text_files_dir).mkdir(parents=True, exist_ok=True)  # Könyvtár létrehozása, ha még nem létezik
    
    # Csak akkor folytatjuk, ha a könyvtár üres
    if not os.listdir(text_files_dir):  # Ellenőrizzük, hogy a könyvtár üres-e
        for idx, func in enumerate(dataset['func']):
            with open(f"{text_files_dir}/func_{idx}.c", "w", encoding="utf-8") as f:
                f.write(func)
        print("Függvények mentve.")
    else:
        print("A könyvtár nem üres, fájlok nem lettek létrehozva.")

def train_or_load_tokenizer(text_files_dir, tokenizer_path, vocab_size=30522):
    # Ellenőrizzük, hogy léteznek-e már a tokenizer fájlok
    if not Path(f"{tokenizer_path}/vocab.json").exists() or not Path(f"{tokenizer_path}/merges.txt").exists():
        tokenizer = ByteLevelBPETokenizer()

        # Összegyűjtjük a .c fájlok teljes elérési útjait
        file_paths = [str(file_path) for file_path in Path(text_files_dir).glob("*.c")]

        # Ellenőrizzük, hogy vannak-e fájlok a listában
        if not file_paths:
            raise Exception("Nincs .c fájl a megadott könyvtárban.")

        tokenizer.train(files=file_paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        tokenizer.save_model(tokenizer_path)
        print("Tokenizer betanítva és elmentve.")
    else:
        tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_path}/vocab.json",
            f"{tokenizer_path}/merges.txt",
        )
        print("Tokenizer betöltve.")

    return tokenizer

def split_and_expand_dataframe(df, text_column, target_column, max_length):
    new_rows = []
    for index, row in df.iterrows():
        text = row[text_column]
        target = row[target_column]

        while len(text) > max_length:
            new_rows.append({text_column: text[:max_length], target_column: target})
            text = text[max_length:]

        new_rows.append({text_column: text, target_column: target})
    
    new_df = pd.DataFrame(new_rows)
    return pd.concat([df, new_df]).reset_index(drop=True)

Train_dataset = pd.read_parquet('./data/codexglue-train-dataset.parquet')
#Train_dataset = split_and_expand_dataframe(Train_dataset, 'func', 'target', 512)


Validation_dataset = pd.read_parquet('./data/codexglue-validation-dataset.parquet')
Validation_dataset = Validation_dataset[Validation_dataset['func'].str.len() < 512]
#Validation_dataset = split_and_expand_dataframe(Train_dataset, 'func', 'target', 512)

Test_dataset = pd.read_parquet('./data/codexglue-test-dataset.parquet')
Test_dataset = Test_dataset[Test_dataset['func'].str.len() < 512]


save_functions_to_files(Train_dataset, 'c_sources')
tokenizer = train_or_load_tokenizer('c_sources', "custom_tokenizer")
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>", length=512)


Train_dataset = Train_dataset[Train_dataset['func'].str.len() < 512]


x_train = list(Train_dataset['func'])
y_train = list(Train_dataset['target'])

x_val = list(Validation_dataset['func'])
y_val = list(Validation_dataset['target'])

x_test = list(Test_dataset['func'])
y_test = list(Test_dataset['target'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CustomDataset(x_train, y_train, tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_hidden_units = model.config.hidden_size
num_classes = 2  # binary classification
classifier_head = nn.Sequential(
    nn.Linear(num_hidden_units, 512), 
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, num_classes)
)

for param in model.classifier.parameters():
    param.requires_grad = True

model.classifier = classifier_head




optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 1

model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.eval()
val_dataset = CustomDataset(x_val, y_val, tokenizer, max_len=512)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc='Validation'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Validation Precision: {precision * 100:.2f}%')
print(f'Validation Recall: {recall * 100:.2f}%')
print(f'Validation F1: {f1 * 100:.2f}%')

cm = confusion_matrix(true_labels, predictions)

print("Confusion matrix:")
print(cm)