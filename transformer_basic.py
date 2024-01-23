import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import random_split

torch.set_printoptions(sci_mode=False)

# Adatok generálása
num_jobs = 10
num_machines = 5
num_samples = 32 * 500  # A minták száma; igazítható
num_epochs = 3  # Epochok száma; igazítható
batch_size = 32  # A batch mérete; igazítható

def generate_random_machine_job_matrix(num_jobs, num_machines):
    return np.random.randint(20, 100, (num_jobs, num_machines))

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


import numpy as np
import itertools
import random
import torch
from torch.utils.data import TensorDataset
# Gép-job mátrix generálása
machine_job_matrix = np.random.randint(0, 100, (num_jobs, num_machines))

def generate_flowshop_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        # Véletlenszerű job sorrend generálása
        job_order = list(range(num_jobs))
        random.shuffle(job_order)
        # Makespan számítása
        makespan = calculate_makespan(machine_job_matrix, job_order)
        # Adat hozzáadása a datasethez
        dataset.append((makespan, job_order))
    return dataset

dataset = generate_flowshop_dataset(num_samples)

class FlowShopTransformer(nn.Module):
    def __init__(self, num_jobs, embedding_dim, dim_model, num_heads, num_layers, num_positions):
        super(FlowShopTransformer, self).__init__()
        self.input_embedding  = nn.Linear(1, embedding_dim)
        self.input_embedding2  = nn.Linear(embedding_dim, embedding_dim * num_jobs)
        self.output_embedding = nn.Embedding(num_jobs, embedding_dim)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(dim_model, num_jobs)  # A kimeneti dimenzió num_jobs legyen

    def forward(self, src, tgt=None):
        src = self.input_embedding(src)
        src = self.input_embedding2(src)
        src = src.reshape(batch_size, num_jobs, 32)
        if tgt is None:
            # Kezdeti target, pl. start token
            tgt = torch.zeros((src.size(0), num_jobs), dtype=torch.long, device=src.device)

        tgt_embeding = self.output_embedding(tgt)
        tgt_embeding = tgt_embeding.view(batch_size, num_jobs, 32)
        output = self.transformer(src, tgt_embeding)

        output = self.output_layer(output)
        return output

# Modell példányosítása
# num_tokens, dim_model, num_heads, num_layers, num_positions értékeit az adott feladathoz kell igazítani
model = FlowShopTransformer(num_jobs=num_jobs, embedding_dim=32, dim_model=32, num_heads=8, num_layers=3, num_positions=num_jobs)



# PyTorch TensorDataset létrehozása
makespans, job_orders = zip(*dataset)
makespans_tensor = torch.tensor(makespans, dtype=torch.float32)
job_orders_tensor = torch.tensor(job_orders, dtype=torch.long)
flowshop_dataset = TensorDataset(makespans_tensor, job_orders_tensor)

# Az eredeti adatkészlet felosztása tanító és teszt adatkészletre
train_size = int(0.8 * len(flowshop_dataset))  # Például a 80%-a a teljes adatkészletnek
test_size = len(flowshop_dataset) - train_size
train_dataset, test_dataset = random_split(flowshop_dataset, [train_size, test_size])


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
# DataLoader létrehozása a tanító adatokhoz

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss function és optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate; igazítható

# Betanító ciklus

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for makespan, job_order in train_loader:
        # Nullázd a gradienseket
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(makespan.unsqueeze(1).float())
        outputs = outputs.view(-1, num_jobs)  # A kimenetet [batch_size * num_jobs, num_jobs]-re alakítjuk
        job_order = job_order.view(-1)  # A célt [batch_size * num_jobs]-re alakítjuk
        
        # Számítsd ki a veszteséget
        loss = loss_fn(outputs, job_order)
        total_loss += loss.item()

        # Backward pass és optimálás
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


# Teszt DataLoader létrehozása
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Tesztelési ciklus módosítása
def test_model_with_output(model, test_loader, num_jobs):
    model.eval()  # Modell tesztelési módba helyezése
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Ne számítsunk gradienset
        for makespan, job_order in test_loader:
            # Nullákkal feltöltött job_order tensor létrehozása
            dummy_job_order = torch.zeros(makespan.size(0), num_jobs, dtype=torch.long)

            outputs = model(makespan.unsqueeze(1).float())
            outputs = outputs.view(-1, num_jobs)
            predicted = torch.argmax(outputs, dim=-1)
            total_correct += (predicted == job_order.view(-1)).sum().item()
            total_samples += job_order.size(0)

            # Kiírjuk a makespan értékeket és a hozzájuk tartozó javasolt sorrendeket
            for i in range(makespan.size(0)):
                print(f"Makespan: {makespan[i].item()}, Javasolt sorrend: {predicted.view(makespan.size(0), -1)[i].numpy()}")

    accuracy = total_correct / total_samples
    print(f"\nÖsszesített teszt pontosság: {accuracy * 100:.2f}%")

# A modell tesztelése a módosított tesztelési ciklussal
test_model_with_output(model, test_loader, num_jobs)
