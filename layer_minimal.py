import torch.nn as nn
import kernet.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from kernet.layers.loss import *
import random
from kernet.layers.klinear import kLinear

seed = 123
# make deterministic
# utils.make_deterministic
sample_size = 1000
np.random.seed(seed)
x = np.random.rand(sample_size, 32)
# make data
n_classes = 10
# balanced = False
balanced = True
sample_size = 1000
if balanced:
    # random, but balanced, labels
    if sample_size % n_classes:
        raise('n_classes does not divide sample_size')
    n_examples = int(sample_size / n_classes)
    y = []
    for i in range(n_classes):
        y += [i] * n_examples
    y = np.random.permutation(y)
else:
    # random labels
    y = np.random.randint(0, n_classes, (sample_size, ))
torch.manual_seed(seed)
# specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.save("x.npy", x)
np.save("y.npy", y)
x = torch.from_numpy(x).to(device, dtype=torch.float32)
y = torch.from_numpy(y)
ylong = y.to(dtype=torch.long)
y_numpy = y.detach().numpy()

# make model
net_head = nn.Sequential(
    nn.Linear(32, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
).to(device)
net_tail = kLinear(in_features=2, out_features=n_classes, kernel='nn_tanh', evaluation='direct').to(device)

np.save("w2.npy", net_tail.linear.weight.detach().numpy())
np.save("b2.npy", net_tail.linear.bias.detach().numpy())

phi = net_tail.phi
net = nn.Sequential(net_head, net_tail)

# %% Train head
optimizer = torch.optim.Adam(params=net_head.parameters(), lr=1e-3)
loss_fn = srs_contrastive_neo(phi, n_classes)
# loss_fn = srs_nmse_neo(phi, n_classes)
np.save("w0.npy", net_head[0].weight.detach().numpy())
np.save("b0.npy", net_head[0].bias.detach().numpy())
np.save("w1.npy", net_head[2].weight.detach().numpy())
np.save("b1.npy", net_head[2].bias.detach().numpy())
for i in range(15000):
    optimizer.zero_grad()
    repr = net_head(x)
    loss = -loss_fn(repr, ylong)
    # loss = loss_fn(repr, ylong)
    loss.backward()
    optimizer.step()
    
    # save intermediate visualizations
    if (i % 100) == 0:
        print(i, -loss.item()) # unwrap loss to show actual value
        new_repr = phi(net_head(x)).detach().cpu().numpy()
        plt.scatter(new_repr[:,0],new_repr[:,1],c=y_numpy)
        plt.title(f"mdlr repr., epoch {i}")
        plt.show()
        
# %% Train tail
activations = net_head(x).detach()
tail_optimizer = torch.optim.Adam(params=net_tail.parameters(), lr=1e-3)
tail_loss_fn = torch.nn.CrossEntropyLoss()



for i in range(15000):
    tail_optimizer.zero_grad()
    loss = tail_loss_fn(net_tail(activations), ylong)
    loss.backward()
    tail_optimizer.step()
    print(i, loss.item())

# get final accuracy
_, pred = torch.max(net(x), dim=1)
print(f"Final acc (%): {(1 - torch.mean((pred != y).to(torch.float)).item()) * 100:.3f}")

# %% Train composite model using full backprop
net_optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
net_loss_fn = torch.nn.CrossEntropyLoss()
for i in range(5000):
    net_optimizer.zero_grad()
    loss = net_loss_fn(net(x), ylong)
    loss.backward()
    net_optimizer.step()

    if (i % 100) == 0:
        print(i, loss.item())
        net_repr = phi(net_head(x)).detach().cpu().numpy()
        plt.scatter(net_repr[:,0],net_repr[:,1],c=y_numpy)
        plt.title(f"e2e repr., epoch {i}")
        plt.show()
        

# get final accuracy
_, pred = torch.max(net(x), dim=1)
print(f"Final acc (%): {(1 - torch.mean((pred != y).to(torch.float)).item()) * 100:.3f}")
