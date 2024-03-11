import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix

Train_dataset = pd.read_parquet('./data/codexglue-train-dataset.parquet')
Train_dataset = Train_dataset[Train_dataset['func'].str.len() < 512]

Validation_dataset = pd.read_parquet('./data/codexglue-validation-dataset.parquet')
Validation_dataset = Validation_dataset[Validation_dataset['func'].str.len() < 512]

Test_dataset = pd.read_parquet('./data/codexglue-test-dataset.parquet')
Test_dataset = Test_dataset[Test_dataset['func'].str.len() < 512]


x_train = list(Train_dataset['func'])
y_train = list(Train_dataset['target'])

x_val = list(Validation_dataset['func'])
y_val = list(Validation_dataset['target'])

x_test = list(Test_dataset['func'])
y_test = list(Test_dataset['target'])

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')

# Lefagyaszt minden réteget a klasszifikációs réteg kivételével
#for name, param in model.named_parameters():
#    if 'classifier' not in name: # classifier réteg kivételével minden más
#        param.requires_grad = False

class PreEncodedCustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = [int(label) for label in labels]
        self.encodings = [tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt') for text in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = {key: val.squeeze() for key, val in self.encodings[idx].items()}
        encoding['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding

train_dataset = PreEncodedCustomDataset(x_train, y_train, tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = PreEncodedCustomDataset(x_val, y_val, tokenizer, max_len=512)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_hidden_units = model.config.hidden_size
num_classes = 2  # binary classification
classifier_head = nn.Sequential(
    nn.Linear(num_hidden_units, 128), 
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, num_classes)
)

#model.classifier = classifier_head


optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 1

# RobertaForSequenceClassification() and BertForSequenceClassification() has a built in classifier layer
#model.to(device)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    fp16=True,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    optim="adafactor",
    warmup_steps=500,
    #resume_from_checkpoint='./results/checkpoint-500',
    save_steps=2000000
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)


trainer.train()
#trainer.evaluate()

predictions = []
true_labels = []

test_dataset = PreEncodedCustomDataset(x_test, y_test, tokenizer, max_len=512)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Validation'):
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