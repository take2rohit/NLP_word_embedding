#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SkipGramLanguageModeler
import pickle
import numpy as np
import matplotlib.pyplot as plt
from helper import reduce_to_k_dim, plot_embeddings
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(device)


# In[2]:


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


# In[3]:


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

iterator = iter(train_loader)
x, y = iterator.next()
(x.shape, y.shape)


# In[4]:


EMBEDDING_DIM = 300
net = SkipGramLanguageModeler(len(vocab), EMBEDDING_DIM)
net.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters())


# In[5]:


losses = []

iteration = 10001
k = 0
vocab_ind_np = torch.tensor(list(ix_to_word.keys())).cuda()
words = ['barrels', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela','free',
         'freedom','jury','favour','capable','capacity','capital', 'county','india','america', 'americans']

for epoch in range(iteration):
    total_loss = 0
    
    for data in train_loader:
        x, y = data
        log_probs = net.forward(x)
        loss = criterion(log_probs, y).to(device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        net.zero_grad()
        
    losses.append(total_loss)
    
    if epoch %3 == 0:
        print('Epoch: {} Loss: {}'.format(epoch, total_loss))
    
    if epoch %50 == 0:
        
        PATH = './saved_model/saved_model_{}.pth'.format(k)
        torch.save(net.state_dict(), PATH)
        print("\n Model Saved! at epoch no.: {}".format(epoch))
        k+=1
        
        embedding_matrix = net.embedding_outputs(vocab_ind_np).cpu().detach().numpy()
        embedding_matrix_reduced = reduce_to_k_dim(embedding_matrix, k=2)
        plot_embeddings(embedding_matrix_reduced, word_to_ix, words, epoch)
        
print('\n Done Training')


# In[ ]:




