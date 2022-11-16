import torch, torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from densenet_snntorch import  spiking_densenet121
<<<<<<< HEAD

batch_size = 1
=======
import numpy as np
from snntorch import functional as SF
from snntorch import utils
batch_size = 128
>>>>>>> cb87bf32a2fed9533b3704a3e6731aa01c9777bc
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
loss_fn = SF.ce_rate_loss() 
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

num_epochs = 10
num_steps = 10

loss_hist = []
acc_hist = []
torch.autograd.set_detect_anomaly(True)
# training loop
net.train()
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, 10, data)
        #print(spk_rec[0].shape)
        #spk_rec = spk_rec[0]
        #spk_rec = torch.sum(spk_rec, dim=0)
        #spk_rec = torch.reshape(spk_rec, (128, 10))
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        torch.cuda.empty_cache()

        # print every 25 iterations
        if i % 25 == 0:
          print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item()}")

          # check accuracy on a single batch
          #acc = SF.accuracy_rate(spk_rec, targets)  
          #acc_hist.append(acc)
          #print(f"Accuracy: {acc * 100:.2f}%\n")
        torch.cuda.empty_cache()

        