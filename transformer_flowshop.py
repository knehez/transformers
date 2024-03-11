import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures

import math
import numpy as np
import random
import os

torch.set_printoptions(sci_mode=False)
np.random.seed(42)  # A reprodukálhatóság érdekében
torch.manual_seed(100)
# settings
num_jobs = 20
SOS = num_jobs
EOS = num_jobs + 1
num_tokens = num_jobs + 2
num_machines = 5
num_epochs = 8
batch_size = 256
model_path = 'saved_models/flowshop_model.pth'

if os.path.exists(model_path):
    print("A modell fájl létezik.")
    train_requierd = False
else:
    print("A modell fájl nem létezik.")
    train_requierd = True

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
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


class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        embedding_matrix
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        #self.embedding.weight.data.copy_(embedding_matrix)

        # Opcionális: Ha nem szeretnéd, hogy az embedding súlyai módosuljanak a tanítás során
        #self.embedding.weight.requires_grad = False
        
        self.transformer = nn.Transformer(dim_feedforward=2048,
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, source, target, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # source size must be (batch_size, src sequence length)
        # source size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        source = self.embedding(source) * math.sqrt(self.dim_model)
        target = self.embedding(target) * math.sqrt(self.dim_model)
        source = self.positional_encoder(source)
        target = self.positional_encoder(target)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        source = source.permute(1,0,2)
        target = target.permute(1,0,2)
        if tgt_mask == None:
            tgt_mask=self.get_tgt_mask(target.shape[0])
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(source, target, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
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

def calculate_makespan(machine_job_matrix, job_order):
    num_jobs, num_machines = machine_job_matrix.shape
    job_end_times = np.zeros((num_jobs, num_machines))

    # Az első munka befejezési idejének kiszámítása minden gépen
    for m in range(num_machines):
        job_end_times[0, m] = (job_end_times[0, m-1] if m > 0 else 0) + machine_job_matrix[job_order[0], m]

    # További munkák befejezési idejének kiszámítása
    for idx in range(1, num_jobs):
        for m in range(num_machines):
            prev_job_end = job_end_times[idx-1, m]
            prev_machine_end = job_end_times[idx, m-1] if m > 0 else 0
            job_end_times[idx, m] = max(prev_job_end, prev_machine_end) + machine_job_matrix[job_order[idx], m]

    return job_end_times[-1, -1]

# machine-job matrix
machine_job_matrix = [
    54, 83, 15, 71, 77, 36, 53, 38, 27, 87, 76, 91, 14, 29, 12, 77, 32, 87, 68, 94,
    79,  3, 11, 99, 56, 70, 99, 60,  5, 56,  3, 61, 73, 75, 47, 14, 21, 86,  5, 77,
    16, 89, 49, 15, 89, 45, 60, 23, 57, 64,  7,  1, 63, 41, 63, 47, 26, 75, 77, 40,
    66, 58, 31, 68, 78, 91, 13, 59, 49, 85, 85,  9, 39, 41, 56, 40, 54, 77, 51, 31,
    58, 56, 20, 85, 53, 35, 53, 41, 69, 13, 86, 72,  8, 49, 47, 87, 58, 18, 68, 28
]
# converting to num_jobs x num_machines NumPy array
machine_job_matrix = np.array(machine_job_matrix).reshape(num_jobs, num_machines)

poly = PolynomialFeatures(degree=6, include_bias=False)

# Alkalmazzuk a polinomiális transzformációt a mátrixon
embedding_matrix = poly.fit_transform(machine_job_matrix)

embedding_matrix = embedding_matrix[:, :256]

sos_column = np.ones((embedding_matrix.shape[1], 1)).T  # Minden munkához 1-es érték az SOS-hez
eos_column = np.zeros((embedding_matrix.shape[1], 1)).T

embedding_matrix = np.concatenate([embedding_matrix, sos_column, eos_column])

from sklearn.preprocessing import StandardScaler

# Inicializáljuk a StandardScaler-t
scaler = StandardScaler()

# Alkalmazzuk a scaler-t az embedding mátrixra
embedding_matrix = scaler.fit_transform(embedding_matrix)

def generate_flowshop_dataset(num_samples):
    dataset = []
    
    SOS_token = np.array([SOS])
    EOS_token = np.array([EOS])

    min_makespan = 1000000
    while len(dataset) < num_samples:
        # random job order
        job_order_1 = list(range(num_jobs))
        random.shuffle(job_order_1)
        makespan_1 = calculate_makespan(machine_job_matrix, job_order_1)
        counter = 0
        while True:
            job_order_tmp = list(range(num_jobs))
            #i = random.randint(0, num_jobs - 1)
            #j = random.randint(0, num_jobs - 1)
            #job_order_tmp[i], job_order_tmp[j] = job_order_tmp[j], job_order_tmp[i]
            random.shuffle(job_order_tmp)
            makespan_2 = calculate_makespan(machine_job_matrix, job_order_tmp)
            if makespan_2 < makespan_1:
                job_order_1 = np.concatenate((SOS_token, job_order_1, EOS_token))
                job_order_tmp = np.concatenate((SOS_token, job_order_tmp, EOS_token))
                dataset.append([job_order_1, job_order_tmp])
                break    
            counter += 1
            if counter == 100:
                break    
        if min_makespan > makespan_1:
            min_makespan = makespan_1
    print(f"min makespan:  {min_makespan}")
    return dataset

def generate_flowshop_testdata(num_samples):
    dataset = []
    
    SOS_token = np.array([SOS])
    EOS_token = np.array([EOS])

    while len(dataset) < num_samples:
        job_order = list(range(num_jobs))
        random.shuffle(job_order)

        job_order = np.concatenate((SOS_token, job_order, EOS_token))
        job_order = torch.tensor([job_order], dtype=torch.long, device=device)
        dataset.append(job_order)

    return dataset

def batchify_data(data, batch_size=batch_size, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches

# generating dataset
if train_requierd == True:
    train_data = generate_flowshop_dataset(10000)
    train_dataloader = batchify_data(train_data)

val_data = generate_flowshop_dataset(batch_size * 2)
val_dataloader = batchify_data(val_data)

test_data = generate_flowshop_dataset(batch_size * 2)
test_dataloader = batchify_data(test_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=num_tokens, dim_model=256, num_heads=32, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.1,
    embedding_matrix=torch.Tensor(embedding_matrix)
).to(device)

opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

scheduler1 = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

if train_requierd == False:
    model.load_state_dict(torch.load(model_path))

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
    
    scheduler1.step()
    scheduler2.step()
    
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list

def predict(model, input_sequence, max_length=num_tokens):
    model.eval()
    
    y_input = torch.tensor([[SOS]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        k = random.randint(1, 2) # simulate generative network
        next_item = pred.topk(k)[1].view(-1)[-1].item() # num with highest probability
        if next_item == EOS:
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS:
            break

    return y_input.view(-1).tolist()

if train_requierd:
    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, num_epochs)
    # Modell mentése
    torch.save(model.state_dict(), model_path)

job_order_1 = list(range(num_jobs))
random.shuffle(job_order_1)
example = torch.tensor([np.concatenate((np.array([SOS]), job_order_1, np.array([EOS])))], dtype=torch.long, device=device)

for i in range(100): # continous generation from a starting sequence
    result = predict(model, example)
    if len(result) != num_tokens:
        print(f"Invalid result: {result[1:-1]}")
        break
    input = example.view(-1).tolist()[1:-1]
    valid = len(result[1:-1]) == len(set(result[1:-1]))

    input_makespan = calculate_makespan(machine_job_matrix, input)
    target_makespan = calculate_makespan(machine_job_matrix, result[1:-1])
    if input_makespan <= target_makespan:
        print("Not better")
        continue
    print(f"Input: {input} - Makespan: {input_makespan}")
    print(f"Prediction ({valid}): {result[1:-1]} - Makespan: {target_makespan}")
    print()
    example = torch.tensor([result], dtype=torch.long, device=device)
  
#for idx, example in enumerate(generate_flowshop_testdata(10)):
#    result = predict(model, example)
#    input = example.view(-1).tolist()[1:-1]
#    print(f"Input: {input} - Makespan: {calculate_makespan(machine_job_matrix, input)}")
#    print(f"Prediction: {result[1:-1]} - Makespan: {calculate_makespan(machine_job_matrix, result[1:-1])}")
#    print()

