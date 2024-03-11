import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# Adatkészlet létrehozása
def generate_encoded_tensors(batch_size, num_symbols, device='cuda'):
    numbers_batch = []
    results_batch = []

    for _ in range(batch_size):
        # Két háromjegyű véletlenszám generálása
        num1 = random.randint(100, 999)
        num2 = random.randint(100, 999)

        # A számok összege
        sum_result = num1 + num2

        encoded_numbers = [num1] + [num2]
        encoded_result = [sum_result]

        # Normalizálás a megadott num_symbols hosszára, 14-es értékkel feltöltve
        encoded_numbers += [PAD] * (num_symbols - len(encoded_numbers))
        #encoded_result += [PAD] * (num_symbols - len(encoded_result))

        # Tenzorok hozzáadása a batch-ekhez
        numbers_batch.append(encoded_numbers)
        results_batch.append(encoded_result)

    # Tenzorok létrehozása a batch-ekből
    numbers_tensor = torch.tensor(numbers_batch).to(device)
    result_tensor = torch.tensor(results_batch).to(device)
    
    return numbers_tensor, result_tensor

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
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(2000, d_model)  # +1 az extra "11-es" számért
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.posencoding = PositionalEncoding(d_model, max_len = 5000, dropout=0.1)
        self.decoder = nn.Linear(d_model, 2000)  # +1 az extra "11-es" számért
        self.dim_model = d_model
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.dim_model) # Bemeneti egész számok beágyazása
        src = self.posencoding(src)
        src = self.transformer_encoder(src)  # Transformer encoder rétegen való átadása
        output = self.decoder(src)  # Dekódolás a kimeneti méretre
        return output

# Modell paraméterei
input_dim = 10  # Szókincs mérete (pl. 0-10)
d_model = 256  # Embedding dimenzió
nhead = 4  # Fejek száma az attentionban
dim_feedforward = 256  # Feedforward hálózat dimenziója
dropout = 0.1  # Dropout arány
PAD=1999
device = "cuda" if torch.cuda.is_available() else "cpu"
# Modell, veszteségfüggvény és optimalizáló inicializálása
model = TransformerModel(input_dim, d_model, nhead, dim_feedforward, dropout).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=1999)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adatkészlet generálása
num_samples = 2000
X, Y = generate_encoded_tensors(num_samples, 10)
#Y = Y.view(-1)  # Átalakítás egydimenziósra a CrossEntropyLoss számára

# Tanítás
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    #outputs = outputs.view(-1, 2000)  # Átalakítás a veszteségfüggvény számára
    loss = criterion(outputs, Y.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
# Tesztelés
model.eval()
with torch.no_grad():
    test_X, test_Y = generate_encoded_tensors(10, 10)
    predictions = model(test_X)
    predicted_indices = torch.argmax(predictions, dim=-1)
    print("\nTeszt bemenet:")
    print(test_X)
    print("\nElőrejelzések:")
    print(predicted_indices)
    print("\nValós kimenet:")
    print(test_Y)