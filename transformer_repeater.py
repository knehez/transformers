import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.set_printoptions(sci_mode=False)

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_layers):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(d_model=dim_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        return self.output_layer(output)

# Modell parameters
num_tokens = 91  # from 0 to 90, including 0 (maximum 9 * 10 = 90)
dim_model = 512
num_heads = 8
num_layers = 3

model = SimpleTransformerModel(num_tokens, dim_model, num_heads, num_layers)

# settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100
batch_size = 1

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # simulated data: input from 1 to 10, output from 10 to 100
    data = torch.randint(1, 10, (batch_size, 10))
    input = data
    target = data * 10  # multiplied by 10

    # prediction
    output = model(input, input)
    loss = criterion(output.view(-1, num_tokens), target.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# test
model.eval()
with torch.no_grad():
    test_data = torch.randint(1, 10, (1, 10))
    test_input = test_data
    dummy_test_input = torch.zeros(1, 10, dtype=torch.int)
    test_output = model(test_input, dummy_test_input)
    predicted = torch.argmax(test_output, dim=-1)
    print("input:", test_input)
    print("predicted output:", predicted)
