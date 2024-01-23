import torch

# hyperparameters
iterations = 21
learning_rate= 0.5

# regenerate the data and model
x = torch.randn(1000, 32)                      # 1000 instances, with 32 features
wt, bt = torch.randn(32, 1), torch.randn(1)    # parameters of the true model
t = torch.mm(x, wt) + bt 

w = torch.randn(32, 1, requires_grad=True) 
b = torch.randn(1, requires_grad=True)

for i in range(iterations):

    # forward pass
    y = torch.mm(x, w) + b

    # mean-squared-error loss 
    r = t - y
    loss = (r ** 2).mean()

    # backpropagation
    loss.backward()
        
    # print the loss
    print(f'iteration {i: 4}: loss {loss:.4}')
    
    # gradient descent
    w.data = w.data - learning_rate * w.grad.data
    b.data = b.data - learning_rate * b.grad.data
    # -- Note that we don't want the autodiff engine to compute gradients over this part.
    #   by operrating on w.data, we are only changing the values of the tensor not 
    #   remembering a computation graph.

    # delete the gradients
    w.grad.data.zero_()
    b.grad.data.zero_()
    # -- if we don't do this, the gradients are remembered, and any new gradients are added
    #    to the old.   

# show the true model, and the learned model
print()
print('true model: ', wt.data[:4].t(), bt.data)
print('learned model:', w.data[:4].t(), b.data)
    

from torch.optim import Adam

# hyperparameters
iterations = 101
learning_rate= 2.0

# regenerate the data and model
x = torch.randn(1000, 32)                      # 1000 instances, with 32 features
wt, bt = torch.randn(32, 1), torch.randn(1)    # parameters of the true model
t = torch.mm(x, wt) + bt 

w = torch.randn(32, 1, requires_grad=True) 
b = torch.randn(1, requires_grad=True)

# Create the optimizer. It needs to know two things:
# - the learning rate
# - which parameters its responsible for
opt = Adam(lr=learning_rate, params=[w, b])

for i in range(iterations):

    # forward/backward, same as before
    y = torch.mm(x, w) + b
    r = t - y
    loss = (r ** 2).mean()
    
    loss.backward() 
    # -- Note that the optimizer _doesn't_ compute the gradients. We still
    #    do that ourselves. The optimizer takes the gradients, and uses them
    #    to adapt the parameters. 
        
    # print the loss
    if i % 20 == 0:
        print(f'iteration {i: 4}: loss {loss:.4}')
    
    # Perform the gradient descent step
    opt.step() 
    
    # The optimizer can zero the gradients for us 
    # (but we still have to tell it to do so)
    opt.zero_grad()