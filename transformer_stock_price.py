import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

torch.set_printoptions(sci_mode=False)

num_days = 300
np.random.seed(42)

d_model = 32
nhead = 16
num_layers = 4
dropout = 0.0
lr = 0.001
batch_size = 32

# Alap szinuszhullám létrehozása
time = np.arange(0, num_days, 1)
amplitude = 100 
base_line = 50
stock_prices = base_line + amplitude * np.sin(2 * np.pi * time / 100)

# add noise to make it more realistic
noise = np.random.normal(0, 5, num_days)
stock_prices += noise

scaler = StandardScaler()
stock_prices = scaler.fit_transform(np.array(stock_prices).reshape(-1, 1))



device = "cuda" if torch.cuda.is_available() else "cpu"

# Sequence Data Preparation
SEQUENCE_SIZE = 10

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Adatok felosztása train és test részekre 80%-20% arányban
total_size = len(stock_prices) - SEQUENCE_SIZE
train_size = int(total_size * 0.8)

x_train, y_train = to_sequences(SEQUENCE_SIZE, stock_prices[:train_size+SEQUENCE_SIZE])
x_test, y_test = to_sequences(SEQUENCE_SIZE, stock_prices[train_size:])

# Ellenőrizzük a méreteket
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Setup data loaders for batch
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len=5000):
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

class StockPriceTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(1, d_model)
        self.posencoding = PositionalEncoding(dim_model=d_model, dropout=dropout)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)    
        
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.posencoding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

model = StockPriceTransformer(d_model, nhead, num_layers, dropout=dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 1000
early_stop_count = 0
min_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        print("Early stopping!")
        break

    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.squeeze().tolist())

rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")

# collecting data for plotting
predictions = []
ground_truth = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.cpu().squeeze().numpy())
        ground_truth.extend(y_batch.cpu().squeeze().numpy())

# inverse scaler transforms back to the original interval
stock_prices = scaler.inverse_transform(np.array(stock_prices).reshape(-1, 1))
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
ground_truth = scaler.inverse_transform(np.array(ground_truth).reshape(-1, 1))

import matplotlib.pyplot as plt

# drawing
plt.figure(figsize=(14, 7))
plt.plot(stock_prices, label='Stock Price', color='blue')  # Az eredeti részvényárfolyam

test_indices = range(len(stock_prices) - len(predictions), len(stock_prices))

plt.plot(test_indices, predictions, label='Prediction', color='red')
plt.plot(test_indices, ground_truth, label='Ground Truth', color='green')

plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()