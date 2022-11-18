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
batch_size = 128
num_epochs = 10
num_steps = 25
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

beta = 0.5
spike_grad = surrogate.fast_sigmoid()
#  Initialize Network
#num_init_features = 64
#net = nn.Sequential(nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
#                    nn.BatchNorm2d(num_init_features),
#                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                    nn.Flatten(),
#                    nn.Linear(4096, 10),
#                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#                    ).to(device)

net = nn.Sequential(nn.Conv2d(3, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(1600, 10),
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

loss_fn = nn.CrossEntropyLoss().to(device)

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

def print_batch_accuracy(data, targets, train=False):
    output, _ = forward_pass(net,num_steps,data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc 

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
test_acc_hist = []

# training loopnum_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0
training_acc = []
test_acc = []
# Outer training loop
epochs = np.arange(num_epochs)
for epoch in range(num_epochs):
    print("EPOCH : ", epoch)
    iter_counter = 0
    train_batch = iter(train_loader)
    avg_loss = 0.0
    avg_test_loss= 0.0
    avg_acc = 0.0
    avg_test_acc = 0.0
    # Minibatch training loop
    for i, (data, targets) in enumerate(train_batch):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = forward_pass(net,num_steps,data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        avg_loss += loss_val.item()
        # Store loss history for future plotting
        avg_acc += print_batch_accuracy(data, targets, train=True)
        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = forward_pass(net,num_steps,test_data)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss_fn(test_mem[step], test_targets)
            avg_test_loss += test_loss.item()
            avg_test_acc += print_batch_accuracy(test_data, test_targets, train=False)

            # print every 25 iterations
            if i % 10 == 0:
                acc = print_batch_accuracy(test_data, test_targets, train=False)
                print(f"Epoch {epoch}, Iteration {i} \nSingle mini batch test acc: {acc*100:.2f}%")
 
    loss_hist.append(avg_loss/len(train_batch))
    training_acc.append(avg_acc/len(train_batch))
    test_loss_hist.append(avg_test_loss/len(train_batch))
    avg_acc = avg_test_acc/len(train_batch)
    test_acc_hist.append(avg_acc)
    print(f"Epoch {epoch}, Iteration {i} \nAvg train acc: {avg_acc*100:.2f}%")
          
plt.figure() 
plt.plot(epochs, loss_hist, label="Training loss")
plt.plot(epochs, test_loss_hist, label="Validation loss")
plt.savefig("loss.pdf")
