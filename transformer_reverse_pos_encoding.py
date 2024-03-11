import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math
import random

torch.set_printoptions(sci_mode=False)

# Modell parameters
num_epochs = 1000
batch_size = 8

dim_model = 128
num_heads = 8
num_layers = 3

num_symbols = 10 # --> 10 digits
SOS = num_symbols
EOS = num_symbols + 1
PLUS = num_symbols + 2
PAD = num_symbols + 3
SOS_token = torch.tensor([[SOS]] * batch_size)
EOS_token = torch.tensor([[EOS]] * batch_size)
num_tokens = num_symbols + 4  # num_symbols + <eos> and <sos> tokens

def calc_target(x):
    """
    Calculate the target from the input.
    """
    y = copy.deepcopy(x)
    #y[0] = reversed(y[0])
    return y

def generate_encoded_tensors(batch_size, num_symbols, device='cpu'):
    numbers_batch = []
    results_batch = []

    for _ in range(batch_size):
        # Két háromjegyű véletlenszám generálása
        num1 = random.randint(100, 999)
        num2 = random.randint(100, 999)

        # A számok összege
        sum_result = num1 + num2

        # A számok és a '+' jelet 13-as számmal kódolva
        encoded_numbers = [int(digit) for digit in str(num1)] + [PLUS] + [int(digit) for digit in str(num2)]

        # Az összeg számjegyenként kódolva
        encoded_result = [int(digit) for digit in str(sum_result)]

        # Normalizálás a megadott num_symbols hosszára, 14-es értékkel feltöltve
        encoded_numbers += [PAD] * (num_symbols - len(encoded_numbers))
        encoded_result += [PAD] * (num_symbols - len(encoded_result))

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

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(dim_model, 1)
        self.posencoding = PositionalEncoding(dim_model, max_len = batch_size, dropout=0.3)
        self.dim_model = dim_model
    def forward(self, src, tgt, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.dim_model) 
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model) 
        src = self.posencoding(src)
        tgt = self.posencoding(tgt)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_layer(output)

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleTransformerModel(num_tokens, dim_model, num_heads, num_layers).to(device)

# settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    input, target = generate_encoded_tensors(batch_size, num_symbols)

    input = torch.cat((SOS_token, input, EOS_token), dim=1).to(device)
    target = torch.cat((SOS_token, target, EOS_token), dim=1).to(device)
    
    #print(input, target)
    
    y = target[:,:-1]
    target_expected = target[:,1:]
    
    sequence_length = y.size(1)
    tgt_mask = model.get_tgt_mask(sequence_length)

    # prediction
    output = model(input, y, tgt_mask=tgt_mask)
    output = output.permute(0, 2, 1)    
    loss = criterion(output, target_expected)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# test
model.eval()
with torch.no_grad():
    
    test_input, test_expected = generate_encoded_tensors(1,num_symbols, device)

    test_target = torch.tensor([[SOS]]).to(device)
    for i in range(num_tokens):
        tgt_mask = model.get_tgt_mask(test_target.size(1))
        test_output = model(test_input, test_target, tgt_mask)
        next_item = test_output.topk(1)[1].view(-1)[-1].item()
        next_item = torch.tensor([[next_item]]).to(device)
        test_target = torch.cat((test_target, next_item), dim=1).to(device)

    print(f'input            : {test_input}')
    print(f'predicted output : {test_target[0][1:num_tokens]}') # try to cut <sos> and <eos>
    print(f'expected output  : {test_expected}')
    
