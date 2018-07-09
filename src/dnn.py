#! /usr/bin/env python

# ==================================== 
# Fully Connected Feedforward Network
# Author: Aman Hussain
# ====================================

# Import Libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# set GPU device
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 5 
bs = 100
lr = 0.001
input_sz = 28*28
output_sz = 10
hidden_sz = 500


# Load Data
trnsfrms = transforms.Compose([transforms.ToTensor()])
trn_set = torchvision.datasets.MNIST(root="../data/", train=True, transform=trnsfrms, download=True)
test_set = torchvision.datasets.MNIST(root="../data/", train=False, transform=trnsfrms, download=True)

trn_ldr = torch.utils.data.DataLoader(dataset=trn_set, batch_size=bs, shuffle=True)
test_ldr = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)


# Build Model
class DNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


# Training
def train():
    md = DNN(input_sz, hidden_sz, output_sz)
    md = md.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(md.parameters(), lr)

    for epoch in range(epochs):
        for i, (img, lbl) in enumerate(trn_ldr):
            # send input to gpu
            img = img.reshape(-1, input_sz)
            img, lbl = img.to(dev), lbl.to(dev)

            # forward
            out = md(img)
            loss = criterion(out, lbl)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print status
            if (i%100) == 99:
                print("Epoch:{} Iterations:{} Loss:{}".format(epoch+1, i+1, loss.item()))
    # Save
    torch.save(md.state_dict(), "../models/dnn.ckpt")


# Testing83
def test(md, report):
    with torch.no_grad():
        correct = 0
        total = 0
        for img, lbl in test_ldr:
            img = img.reshape(-1, input_sz)
            img, lbl = img.to(dev), lbl.to(dev)
            out = md(img)
            _, y_pred = torch.max(out.data, 1)
            total += lbl.shape[0]
            correct += (y_pred == lbl).sum().item()
        print("{} Test Accuracy:{}".format(report, 100 * correct/total))


# Load
md1 = DNN(input_sz, hidden_sz, output_sz)
md1.to(dev)

# Test
test(md1, "Untrained")
md1.load_state_dict(torch.load("../models/dnn.ckpt"))
test(md1, "Trained")