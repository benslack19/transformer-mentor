"""Script for running bigram model."""

# left off at in Karpathy video

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters ------------------------
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000  # what's this
eval_interval = 300  # how frequently to evaluate the loss
eval_iters = 200  # to document losses
learning_rate = 1e-2

# Check if CUDA is available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


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


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # learnable parameter, starts off as random
        # why the shape of `nn.Embedding`?
        # first vocab_size represents the input, need a row for each input token
        # second vocab_size is for the output for every possible next token, (logits: a logit for each possible next token)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        # for each idx, it is plucking out the idx-th row out of that embedding table
        logits = self.token_embedding_table(idx)  # (B,T,C)

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
            # get the predictions; `loss` is ignored because we're not using it here
            logits, loss = self(
                idx
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


model = BigramLanguageModel(vocab_size)
# why is this needed? helps ensure that the model (weights of embedding table) and tensor are in the same place
# (e.g. model may be on GPU, input tensor inadvertently on CPU)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
# i here is a step or iteration (e.g. training step or training iteration)
# an "epoch" is a full pass through the entire dataset, whereas this processes a batch
for i in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # generate on the device (cpu/cuda)
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
