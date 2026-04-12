after feedforward layers, 1:26:42
interperse communication with computation

transformer has blocks of attention + MLP that are replicated

1:28:10: output still doesn't look great, problem is that we're now starting to get a pretty deep neural net
deep neural nets suffer from optimization issues.. he thinks we're running into.
two optimization steps with depth....
1. Skip connection
- skip connections or residual connections; from paper: deep residual learning for image recognition
- still have transformation but you take features and add to it
- project back to original pathway via addition. it can be useful for how backpropagation flows. at initialization, it's like a backpropagation superhighway. addition distributes gradients equally along forks.
- `x = x + some_transformation(x)`  why is it called "residual"? the transformation only needs to learn the difference (the residual) from the original input rather than the full mapping. if the transformation learned nothing, the output would just be `x` unchanged. the network can "skip" the transformation. it's called skip because the original signal literally skips over the transformation block and gets added back at the end. in transfomers, you see this twice:
  - `x = x + self_attention(x)`. (fork off and do some communication then come back)
  - `x = x + feedforward(x)`
- in the beginning, the residual blocks (where the transformation happens) doesn't contribute much to backpropagation but during training, they come online and start to contribute.
- the point - it helps a lot with optimization.


nn.Linear’s parameters are (input_size, output_size) correct? so if it’s (2, 1) would it be like y = Ax + Bz ?
(yes))
What would (2, 2) look like?
```
y1 = Ax + Bz
y2 = Cx + Dz
```

Second optimization step to deal with deep network: Layernorm (geoffrey hinton paper)
Similar to batchnorm. Batchnorm is normalizing the columns to have 0 mean, unit variance.
Layernorm is normalizing the ROWS to have 0 mean, unit variance. (It's a pytorch function)
In the original paper, layernorm was added after the blocks, but now the "innovation" is to add it before (PreNorm).


1:37:52: scale out the model


1:42:14: start training but it needs GPU to work with his settings

1:43:33: talking about how original paper had encoder and decoder blocks, working via cross-attention
our example had decoder block just generate, unconditioned on anything.
with encoder-decoder example, the decoder is conditioned on the encoder output

the decoder's attention still generates a query, but the keys/values come from the encoder output... they feed into every single block of the decoder


1:46:31: nanoGPT
train.py ... very similar to how we trained, but there's other stuff because of distributed training, checkpoints, etc.
model.py is almost identical, one difference is how the heads are organized (all the heads are treated as a batch dimension now)
they're 4 dimensional tensors now
also uses gelu non-linearity, instead of relu

1:52:00: contextualizing what we learned for working implementation of chatGPT (pre-trained and fine-tuning stage AKA fining-stage..)
changes from document completer to "question answerer"

nano GPT is focused on pretraining

IDEA: make an interactive journal of when code gets added
