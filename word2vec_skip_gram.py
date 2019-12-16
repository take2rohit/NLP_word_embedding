#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(device)


with open('vocabulary.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

with open('bigrams_dataset.pickle', 'rb') as handle:
    bigrams_dataset = pickle.load(handle)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = { i: word for i, word in enumerate(vocab)}

x_data = []
y_data = []
for bigram in bigrams_dataset:
    x_data.append(word_to_ix[bigram[0]])
    y_data.append(word_to_ix[bigram[1]])



class Word_dataset(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data).to(device)
        self.y_data = torch.tensor(y_data).to(device)
        self.length = len(x_data)
        
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.length
    
train_dataset = Word_dataset(x_data, y_data)
train_loader = DataLoader(dataset=train_dataset, batch_size = 2**14+10000, shuffle=True)



class SkipGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

EMBEDDING_DIM = 300
net = SkipGramLanguageModeler(len(vocab), EMBEDDING_DIM)
net.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters())


losses = []

iteration = 10000

print("Training started on: ", device_name)

for epoch in range(iteration):
    total_loss = 0
    
    for data in train_loader:
        x, y = data
        x, y = Variable(x).to(device), Variable(y).to(device)
        log_probs = net.forward(x)
        loss = criterion(log_probs, y).to(device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        net.zero_grad()
        
    losses.append(total_loss)
    
    if epoch %2 == 0:
        print('Epoch: {} Loss: {}'.format(epoch, total_loss))
    
    if epoch % 50 == 0:
        PATH = './saved_model.pth'
        torch.save(net.state_dict(), PATH)
        print("Model Saved! at epoch no.: {}".format(epoch))