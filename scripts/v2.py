"""Script for running bigram model."""

# left off at in Karpathy video

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters ------------------------
# batch_size = 32  # how many independent sequences will we process in parallel?
# block_size = 8  # what is the maximum context length for predictions?
# after scaling up, FYI Karpathy had an A100 GPU so his training went a lot faster
batch_size = 64
# block_size = 256 # 256 characters of context to predict the 257th
block_size = 64  # bump down to help with training


max_iters = 5000  # bump up max_iters from 3000 > 5000 after doing self-attention
# eval_interval = 300  # how frequently to evaluate the loss
eval_interval = 10  # how frequently to evaluate the loss (in final script it's set to 500, but making it smaller to see some output)

# eval_iters = 200  # to document losses
eval_iters = 10  # to document losses, make it smaller to see some output

# learning_rate = 1e-2
# learning_rate = 1e-3 # make it smaller after self attention
learning_rate = 3e-4  # make it smaller after scaling up
# n_embed = 32
n_embed = 384
n_head = 6  # 384 / 6 means every head is 64 dimensional
n_layer = 6
dropout = 0.2

# Check if CUDA is available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(1337)

PROJECT_DATA = "/workspaces/transformer-mentor/project_data/"
# -------------------------------------------------


with open(Path(PROJECT_DATA) / "input.txt", encoding="utf-8") as file:
    text = file.read()

# all the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)  # global variable, not necessary to be part of BigramLanguageModel object

# tokenize, create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# decoder: take a list of integers, output a string
decode = lambda l: "".join([itos[i] for i in l])

# tokenize the text dataset
# note: specify torch.long for integer
# since pytorch defaults to float32(long = 64-bit integer)
data = torch.tensor(encode(text), dtype=torch.long)

# Split data into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# serving as the dataloader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)  # moving data if warranted
    return x, y


# averages up the loss of multiple batches for both splits
# why is there both model.eval() and model.train()? good habit to think about what
# mode the model is in (some layers will have different behavior at training time or inference mode)
# need to think if we're using dropout layers, batch norm layers, etc.
@torch.no_grad()  # just means that everything inside this function we won't call .backward on
# pytorch can be more memory efficient (doesn't have to store intermediate variables) (good practice to do this if it applies)
def estimate_loss():
    out = {}
    model.eval()  # sets model to evaluation phase (testing, validating, predicting, generating)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # resets model to training phase
    return out


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        # need to do linear regression for each k, q, v
        # since each gets its own learnable weights so they can specialize into those different roles
        # during training
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # tril is not a parameter of the model
        # in pytorch naming conventions this is called a buffer, not a parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout_p = dropout

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        # happens on every forward and backward pass
        self.dropout = nn.Dropout(dropout)  # regularization, added while scaling up the model

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # not communicating with the past makes it a decoder bock
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B, T, T
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # added while scaling up the model; randomly prevent some nodes from communicating
        # dropout from hinton paper: every forward and backward pass shuts off some set of neurons (drops them to 0)
        # trains without them... like training an ensemble of subnetworks
        # then at test time, all subnetworks get merged
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        
        # Use optimized scaled_dot_product_attention for efficiency.
        # This replaces manual scaling, masking, softmax, and dropout.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.dropout_p if self.training else 0.0
        )
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    # allows for additional communication
    # e.g. find consonants, vowels... and at certain positions
    # find different things

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # projection added during residual block
        # maps it back to (B, T, n_embed) so that the shapes will be compatible for addition
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all heads in parallel and concatenate all outputs
        # over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)  # need to apply projection - linear transformation of the outcome of that layer
        # "projection" because it's going from one dimensional space to another space
        # why not just do a mechanical reshape using `.view()`
        # the model can learn how to best combine and mix outputs of all the heads before feeding back into the residual stream
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity.

    Each token will "think" on the data individually
    after multi-headed attention has occurred
    """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embed, 4 * n_embed
            ),  # the 4 factor is from attention is all you need paper (growing network on the side)
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer going back to residual pathway
            nn.Dropout(
                dropout
            ),  # added while scaling up the model, add it right before the connection to residual pathway
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # the mean and variance are taken over n_embed (32) numbers (batch and time are a factor?)
        # like a per token transformation
        # it will start as unit mean/variance at initialization but these are learnable parameters
        # optimization will determine that
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # the `x + ` is for residual connection
        x = x + self.sa(self.ln1(x))  # apply layernorm BEFORE it goes into self-attention
        x = (
            x + self.ffwd(self.ln2(x))
        )  # computation is done on all the tokens independently, apply layernorm BEFORE it goes into the feedforward network
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # learnable parameter, starts off as random
        # why the shape of `nn.Embedding`?
        # first vocab_size represents the input, need a row for each input token
        # changed second vocab_size to n_embed, won't give us logits directly

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # add in self-attention head
        # self.sa_head = Head(n_embed)

        # with MH attention
        # get 4 communication channels
        # self.sa_heads = MultiHeadAttention(4, n_embed // 4)  # i.e. 4 heads of 8-dimensional self-attention

        # # MLP layer
        # self.ffwd = FeedForward(n_embed)

        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     nn.LayerNorm(n_embed) # one more layernorm at end of transformer and before linear layer
        # )
        # when scaling up the model, make some cosmetic changes to make it nicer
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm

        # add in another layer of processing (pedagogically, this is added before positional encoding)
        # lm_head = language model head
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        # for each idx, it is plucking out the idx-th row out of that embedding table
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)

        # have a representation where token embedding has positional information
        # broadcasting will work out (gets right-aligned, one dimension gets added?)
        x = tok_emb + pos_emb
        # B, T, vocab_size # this C is not same as line above it

        # simplest way to plug in self-attention
        # x = self.sa_heads(x) # apply one head of self-attention

        # x = self.ffwd(x)  # (B, T, C)

        # go through the calculations for each block
        x = self.blocks(x)  # B, T, C

        logits = self.lm_head(x)

        # compute loss function (arguments are predicted output (logits) and actual output (target))
        # how well are we predicting the next character
        # the value at the predicted token index should be a high number
        # this will error because pytorch cross_entropy expects B*T,C
        # loss = F.cross_entropy(logits, targets)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # .view is reshaping the tensor, doesn't change underlying view
            # F.cross_entropy expects (N,C) where N is the number of examples
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # added after doing self attention head
            # need to crop index so it fits within block size
            idx_cond = idx[:, -block_size:]

            # get the predictions; `loss` is ignored because we're not using it here
            logits, loss = self(
                idx_cond
            )  # this will go to the forward function, because `nn.Module` triggers the `__call__` method which internally calls the `forward` function (need to make targets optional)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # B,C, the -1 applies the softmax to the last dimension of the tensor (C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
# why is this needed? helps ensure that the model (weights of embedding table) and tensor are in the same place
# (e.g. model may be on GPU, input tensor inadvertently on CPU)
m = model.to(device)

# print the number of parameters in th emodel
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
# i here is a step or iteration (e.g. training step or training iteration)
# an "epoch" is a full pass through the entire dataset, whereas this processes a batch
for i in range(max_iters):
    print(f"step {i}")  # temporary debug line

    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    # pay attention to what's happening to the train and val loss with each iteration
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # generate on the device (cpu/cuda)
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
print(decode(m.generate(idx=context, max_new_tokens=500)[0].tolist()))
