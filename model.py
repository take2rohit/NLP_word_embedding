import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out)
        return log_probs
    
    def embedding_outputs(self, inputs):
        return self.embeddings(inputs)