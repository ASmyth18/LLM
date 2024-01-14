#!/C:/Users/adams/anaconda3/python.exe

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import mmap
import random
import pickle
import argparse
import os
import sys


os.environ['CUDA_HOME'] = r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3'
os.environ['PATH'] = r';'.join([
    r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin',
    r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/libnvvp',
    os.environ['PATH']
])

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')


# os.environ['CUDA_HOME'] = r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3'  
print(sys.path)


parser = argparse.ArgumentParser(description='ARGPARSER')
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()
print(f'batch size: {args.batch_size}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available() == True
# device = torch.device('cpu')
print(device)

block_size = 384
batch_size = args.batch_size
max_i = 5000
learning_rate = 3e-4
eval_iters = 250
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2


chars = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
    
vocabulary_size = len(chars)


string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])




class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T].expand_as(wei) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nn_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x) # self attention
        x = self.ln1(x + y) # add a norm
        y = self.ffwd(x) # feed forward
        x = self.ln2(x + y) # add a norm
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
         # Initialize an embedding table with self-loop for each token
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        N, T = index.shape
        
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            # If no targets provided, return None for loss
            loss = None
        else:
            # If targets are provided, calculate cross-entropy loss
            N, T, C = logits.shape
            logits = logits.view(N*T, C)
            targets = targets.view(N*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # Generate new tokens using the trained model
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            # Sample the next token based on the computed probabilities
            index_next = torch.multinomial(probs, num_samples=1)
            # Concatenate the new token to the input sequence
            index = torch.cat((index, index_next), dim=1)
        return index
    
print("Loading model parameters...")

with open('model-01.pkl', 'rb') as f:
    model = torch.load(f, map_location=torch.device('cpu'))


m = model.to(device)
print("Loaded Successfully")

while True:
    prompt = input("Prompt:\n")    
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars	= decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')


