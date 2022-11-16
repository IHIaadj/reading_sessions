# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

# dataloader arguments
batch_size = 8
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

train = datasets.CIFAR10(".", train=True, download=True, transform=transform)
test = datasets.CIFAR10(".", train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

beta = 0.9
spike_grad = surrogate.fast_sigmoid()
#  Initialize Network
num_init_features = 64
net = nn.Sequential(nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(num_init_features),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(4096, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)


def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

loss_fn = SF.ce_rate_loss()

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 2
num_steps=1
test_acc_hist = []

# training loop
for epoch in range(num_epochs):

    #avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn, 
    #                        num_steps=num_steps, time_var=False, device=device)
    
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)

        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Train Loss: {loss_val.item()}")

    # Test set accuracy
    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)

    print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")

fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


