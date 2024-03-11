import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.set_printoptions(sci_mode=False)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Modell definiálása
class SimpleTransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, dim_feedforward, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)  # +1 az extra "11-es" számért
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=dim_feedforward)
        self.decoder = nn.Linear(d_model, num_tokens)  # +1 az extra "11-es" számért

    def forward(self, src):
        src = self.embedding(src)  # Bemeneti egész számok beágyazása
        src = self.transformer_encoder(src)  # Transformer encoder rétegen való átadása
        output = self.decoder(src)  # Dekódolás a kimeneti méretre
        return output

# Modell parameters
num_tokens = 10  # from 0 to 90, including 0 (maximum 9 * 10 = 90)
dim_model = 128
num_heads = 8
num_layers = 3

model = SimpleTransformerModel(num_tokens, dim_model, num_heads, num_layers)

# settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
batch_size = 32

data = torch.randint(0, num_tokens, (batch_size, num_tokens))

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # simulated data: input from 1 to 10, output from 10 to 100
    
    input = data
    target = data # multiplied by 10
    #target[0][-1] = 9

    # prediction
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        data = torch.randint(0, num_tokens, (batch_size, num_tokens))
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# test
model.eval()
with torch.no_grad():
    test_data = torch.randint(0, num_tokens, (1, num_tokens))
    test_input = test_data
    dummy_test_input = torch.zeros(1, 10, dtype=torch.int)
    test_output = model(test_input)
    predicted = torch.argmax(test_output, dim=-1)
    print("input:", test_input)
    print("predicted output:", predicted)
