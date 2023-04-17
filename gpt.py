#!//usr/bin/python3
#
#
import sys
import io
import math
import numpy as np
import os
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



''' charachter level tokenizer '''
class Corpus:
    def __init__(self, text):
        self.input_text = text
        self.input_len = len(text)
        self.chars = sorted(list(set(text)))
        self.vocabulary_size = len(self.chars)

    def __repr__(self):
        uc = ''.join(self.chars)
        dlen = len(self.input_text)
        return f"{self.__class__}\n\tlength of data-set: {dlen}\n\tchars: {uc}\n\tvocabulary_size: {self.vocabulary_size}"

    ''' encoder take a string and output a list of integers '''
    def text_encode(self, text):
        str_to_int = { ch:i for i,ch in enumerate(self.chars) }
        enc = lambda s: [str_to_int[c] for c in s] # encoder
        return enc(text)

    ''' decoder take a list of integers, output a string '''
    def text_decode(self, text):
        int_to_str = { i:ch for i,ch in enumerate(self.chars) }
        dec = lambda l: ''.join( [int_to_str[i] for i in l] ) # decoder
        return dec(text)

    def reset_text_data(self, ntext):
        self.input_text = ntext
        self.input_len = len(ntext)
        self.chars = sorted(list(set(ntext)))
        self.vocabulary_size = len(self.chars)



class TextEncoder(Corpus):
    def __init__(self, model, optimizer, args, text):
        try:
          super().__init__(text)
          self.data = None
          self.train_data = None
          self.validation_data = None
          self.batch_size = args['batch_size']  # how many independent sequences will we process in parallel
          self.block_size = args['block_size']  # max context len for predictions
          self.model = model
          self.optimizer = optimizer
          self.device = args['device']
          self.max_iters = args['max_iters']
          self.eval_iters = args['eval_iters']
          self.eval_interval = args['eval_interval']

        except KeyError as key_error:
          print(f'wrong input args: invalid {key_error}')
          raise key_error

    def encode_corpus(self, train_percent=0.9):
        self.data = torch.tensor(self.text_encode(text), dtype=torch.long, device=self.device)

        # split the data into train data and validation data parts
        n = int(train_percent*len(self.data))
        self.train_data = self.data[:n]
        self.validation_data = self.data[n:] # determine how much our model is overfitting

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.validation_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,)) # random offsets into the data set
        x = torch.stack([ data[i:i+self.block_size] for i in ix])      # with block-size=8 end up with  4 x 8 array
        y = torch.stack([ data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    ''' averages the loss over multiple batches and get the average loss '''
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train','val']:
            losses = torch.zeros(self.eval_iters)  # eval_iters ??
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    ''' training method '''
    def train(self):
        for iter in tqdm(range(self.max_iters)):
            print('training')
            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0:
               losses = self.estimate_loss()
               print(f"step {iter}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # sample a batch
            xb, yb = self.get_batch('train')
            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        # Save the model and optimizer state
        torch.save( self.model.state_dict(), f"data/model_gpt_simple.pt")
        torch.save( self.optimizer.state_dict(), f"data/optimizer_gpt_simple.pt")


''' one head of self-attention '''
class Head(nn.Module):
      def __init__(self, head_size, block_size, num_embed, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # a buffer (lower triangular matrix) 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

      def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # copmpute attention scores i.e "affinities"
        weight = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        weight = weight.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        weight = F.softmax(weight, dim=-1) # (B,T,T)
        weight = self.dropout(weight)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T,C)
        out = weight @ v   # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out 


''' mulitple Heads with a linear and a dropout layer '''
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, cfg):
        super().__init__()
        self.heads = \
             nn.ModuleList([Head(head_size, cfg['block_size'], cfg['num_embeddings'], cfg['dropout']) \
                for _ in range(cfg['head_size'])])

        self.proj = nn.Linear(cfg['num_embeddings'], cfg['num_embeddings'])
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat the output from self-attention
        out = self.proj(out)
        return out


''' simple mulit layer preceptron '''
class FeedForward(nn.Module):
    """ simple linear layer followed by non-linerarity """
    def __init__(self, num_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
                      nn.Linear(num_embed, 4 * num_embed),
                      nn.ReLU(),
                      nn.Linear(4 * num_embed, num_embed),  # projection layer
                      nn.Dropout(dropout)
                   )

    def forward(self, x): 
        return self.net(x)


"""
  Transformer block: communication followed by computation
  num_embed: the number of dimension, head_size the number of heads we'd like to use 
"""
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
          head_size_t = cfg['num_embeddings'] // cfg['head_size']
          self.sa = MultiHeadAttention(head_size_t, cfg)  # 4 heads of 8-dimensional self-attention
          self.ffwd = FeedForward(cfg['num_embeddings'], cfg['dropout'])
          self.ln1 = nn.LayerNorm(cfg['num_embeddings'])
          self.ln2 = nn.LayerNorm(cfg['num_embeddings'])
        except KeyError as key_error:
          print(f'wrong input args: invalid {key_error}')
          raise key_error

    def forward(self, x):   # residual connections
         x = x + self.sa( self.ln1(x))
         x = x + self.ffwd( self.ln2(x))
         return x


'''
   language model
'''
class BigramLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        try:
          self.block_size = cfg['block_size']
          self.head_size = cfg['head_size'] # 4 
          self.device = cfg['device']

          # each token directly reads off the logits for the next tokens from a lookup table
          self.token_embedding_table = nn.Embedding(cfg['vocab_size'], cfg['num_embeddings'])
          self.position_embedding_table = nn.Embedding(cfg['block_size'], cfg['num_embeddings'])
          #
          self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg['num_layers'])])

          self.ln_f = nn.LayerNorm(cfg['num_embeddings']) # final layer norm
          self.lm_head = nn.Linear(cfg['num_embeddings'], cfg['vocab_size'])
        except KeyError as key_error:
          print(f'wrong input args: invalid {key_error}')
          raise key_error

    def forward(self, x, targets=None):
        B, T = x.shape
        # x and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(x) # data is (B,T,C)  batch,bytime,bychannel==4,8,65
        positional_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        idx = token_emb + positional_emb # (B,T,C)

        idx = self.blocks(idx)
        idx = self.ln_f(idx)
        logits = self.lm_head(idx) # data is (B,T,vocab_size) 

        if targets is None:
           loss = None
        else:
           # reshaping our data for use by cross_entropy as our loss function
           B, T, C = logits.shape
           logits = logits.view(B*T, C)
           targets = targets.view(B*T)
           # or use pytorch to guess the output size targets = targets.view(-1)
           loss = F.cross_entropy(logits, targets)  # this expect (B,C)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)   -1 gets the last element in the time dimemsion
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



''' read the training text data '''
def read_input_text(fileName=None):
    if not fileName:
       raise ValueError("filename not supplied")
    text_data = None
    try:
      with open(fileName, 'r', encoding='utf-8') as f:
        text_data = f.read()
      return text_data
    except Exception as e:
        raise e

''' set the toch device '''
def set_torch_device():
    return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
           )



if __name__ == '__main__':

   try:
      text = read_input_text('data/tinyshakespeare/input.txt')
      vocabulary_size = len(sorted(list(set(text))))
      device = set_torch_device()
      #
      # hyperparameters
      cfg = {'vocab_size': vocabulary_size,
             'batch_size': 64,
             'block_size': 256,
             'num_embeddings': 384,
             'head_size': 6,
             'eval_interval': 500,
             'max_iters':  5000,
             'eval_iters': 200,
             'num_layers': 6,
             'dropout': 0.2,
             'device': device,
             'learning_rate': 3e-4
            }
      #
      model = BigramLanguageModel(cfg)
      model = model.to(device)

      optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

      tenc = TextEncoder(model, optimizer, cfg, text)
      print(f'TextEncoder: {tenc}')

      # convert the raw text to integers
      tenc.encode_corpus()

      # train on blocks of the data, random level chunks, i.e complex_length, etc
      print(f'block_size: {tenc.train_data[:tenc.block_size+1]}')

      # training method
      tenc.train()

      # generate from the trained model
      context = torch.zeros((1,1), dtype=torch.long, device=device)
      print(tenc.text_decode(m.generate(context, max_new_tokens=800)[0].tolist()))

      sys.exit(0)
   except Exception as er:
      print(traceback.format_exc())
      sys.exit(1)
