import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import random

torch.set_printoptions(sci_mode=False)

# settings
num_tokens = 10

num_epochs = 5
batch_size = 256

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, dim_feedforward, num_layers, dropout):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(num_tokens, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.posencoding = PositionalEncoding(d_model, max_len = batch_size, dropout=dropout)

        #self.decoder = nn.Linear(d_model, num_tokens) 
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_tokens),
        )
        self.dim_model = d_model
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.dim_model) 
        src = self.posencoding(src)
        
        target = self.transformer_encoder(src)
          
        output = self.decoder(target) 
        return output

def generate_dataset(num_samples):
    dataset = []
    
    while len(dataset) < num_samples:
        # random job order
        data = list(range(num_tokens))
        random.shuffle(data)
        target = data[:]
        dataset.append([data, target])
    return dataset

def batchify_data(data, batch_size=batch_size):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches

# generating dataset
full_data = generate_dataset(10000)
test_data = generate_dataset(10)

random_data = batchify_data(full_data)

# Batch-ek számának meghatározása
num_batches = len(random_data)
num_test_batches = int(num_batches * 0.8)
num_val_batches = num_batches - num_test_batches

# Batch-ek felosztása
train_dataloader = random_data[:num_test_batches]
val_dataloader = random_data[num_test_batches:]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerModel(
    num_tokens=num_tokens, d_model=128, num_heads=2, dim_feedforward=256, num_layers=2, dropout=0.1
).to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def calculate_accuracy(input_list, prediction_list):
    # Az egyezések számának meghatározása
    matches = sum(i == p for i, p in zip(input_list, prediction_list))
    # Az egyezési arány százalékban
    accuracy = 100 * matches / len(input_list)
    return accuracy

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y_expected = batch[:, 0], batch[:, 1]
        X, y_expected = torch.tensor(X).to(device), torch.tensor(y_expected).to(device)

        pred = model(X)
    
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def predict(model, input_sequence):
    model.eval()

    pred = model(input_sequence)
    
    predicted_indices = torch.argmax(pred, dim=-1)

    return predicted_indices

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    
    accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y_expected = batch[:, 0], batch[:, 1]
            X, y_expected = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y_expected, dtype=torch.long, device=device)

            pred = model(X)

            loss = loss_fn(pred, y_expected)
            total_loss += loss.item()

            # calculate accuracy
            result = predict(model, X)

            result = result.view(-1)
            y_expected = y_expected.view(-1)

            accuracy += calculate_accuracy(result, y_expected)
    print(f"Accuracy: {accuracy / len(dataloader)}")
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

def compare_vectors(result, expected):
    # ANSI kódok a színekhez
    RED = "\033[31m"
    RESET = "\033[0m"

    result_str = "tensor(["

    for res, exp in zip(result, expected):
        if res != exp:
            result_str += f"{RED}{res:1d}{RESET}, "  # Piros, ha eltérés van
        else:
            result_str += f"{res:1d}, "  # Normál szín, ha egyezik
    
    return result_str

fit(model, opt, loss_fn, train_dataloader, val_dataloader, num_epochs)

sum_accuracy = 0

for i in range(10):
    test = generate_dataset(1)

    test_input, expected = test[0][0], test[0][1]

    example = torch.tensor([test_input], dtype=torch.long, device=device)
    expected = torch.tensor([expected], dtype=torch.long, device=device)

    prediction = predict(model, example)
    
    input = example.view(-1)
    prediction = prediction.view(-1)
    expected = expected.view(-1)

    accuracy = calculate_accuracy(prediction, expected)
    
    print(f"Input:      {input}")
    predicted_str = compare_vectors(prediction, expected)
    print(f"Prediction: {predicted_str}")
    print(f"Expected:   {expected}\n")
    sum_accuracy += accuracy

print(f"\nAccuracy: {sum_accuracy/10}%")

