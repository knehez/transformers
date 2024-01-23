import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

num_days = 200
stock_prices = np.random.rand(num_days) * 100

input_seq_len = 10
output_seq_len = 5
num_samples = num_days - input_seq_len - output_seq_len + 1

src_data = torch.tensor([stock_prices[i:i+input_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()
tgt_data = torch.tensor([stock_prices[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(num_samples)]).unsqueeze(-1).float()

class StockPriceTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output

d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

model = StockPriceTransformer(d_model, nhead, num_layers, dropout=dropout)


epochs = 100
lr = 0.001
batch_size = 16

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i in range(0, num_samples, batch_size):
        src_batch = src_data[i:i+batch_size].transpose(0, 1)
        tgt_batch = tgt_data[i:i+batch_size].transpose(0, 1)
        
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch[:-1])
        loss = criterion(output, tgt_batch[1:])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


src = torch.tensor(stock_prices[-input_seq_len:]).unsqueeze(-1).unsqueeze(1).float()
tgt = torch.zeros(output_seq_len, 1, 1)

with torch.no_grad():
    for i in range(output_seq_len):
        prediction = model(src, tgt[:i+1])
        tgt[i] = prediction[-1]

output = tgt.squeeze().tolist()
print("Next 5 days of stock prices:", output)