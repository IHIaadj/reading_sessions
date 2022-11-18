import torch, torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from densenet_snntorch import  spiking_densenet121
import numpy as np
from snntorch import functional as SF
from snntorch import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# parameters
num_epochs = 10
num_steps = 25
batch_size = 128

# Define a transform
transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

train = datasets.CIFAR10(".", train=True, download=True, transform=transform)
test = datasets.CIFAR10(".", train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

net = spiking_densenet121(3)
print(net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
#loss_fn = SF.ce_rate_loss() 
loss_fn = nn.CrossEntropyLoss()
# SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

def print_batch_accuracy(data, targets, train=False):
    output, _ = forward_pass(net,num_steps,data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc 

# initialize the total loss value
dtype = torch.float

# training loopnum_epochs = 1
test_loss_hist = []
counter = 0
training_acc = []
test_acc = []
test_acc_hist = []

loss_hist = []
acc_hist = []
torch.autograd.set_detect_anomaly(True)
#net.train()
# training loop
for epoch in range(num_epochs):

    print("EPOCH : ", epoch)
    train_batch = iter(train_loader)
    avg_loss = 0.0
    avg_acc = 0.0
    avg_test_acc = 0.0
    avg_test_loss= 0.0

    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        
        net.train()
        spk_rec, mem_rec = forward_pass(net, num_steps, data)
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for j in range(num_steps):
            loss_val += loss_fn(mem_rec[j], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward(retain_graph=True)
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
    test_loss_hist.append(avg_test_loss/len(test_loader))
    avg_acc = avg_test_acc/len(test_loader)
    test_acc_hist.append(avg_acc)
    print(f"Epoch {epoch}, Iteration {i} \nAvg train acc: {avg_acc*100:.2f}%")
