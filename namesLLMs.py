# MAKEMORE: LEARN CHARACTER PATTERNS, THEN INVENT NAMES
#
# This program reads one word per line, learns to predict the next character,
# and repeatedly uses its own predictions to make new words. It does not have
# handwritten rules for names. Training adjusts numbers called parameters so
# that real character sequences become more likely. These are small language
# models: the tokens here are individual characters, rather than word pieces.
#
# READING PYTHON FOR THE FIRST TIME
# Read from top to bottom, but remember: def and class DEFINE things; their
# bodies run later when called. Indentation groups statements into a body.
# A colon starts a body. Parentheses call a function; name=value inside a call
# supplies a named argument. Outside a call, = assigns a value to a name.
# == compares values; != means not equal. A dot accesses something belonging
# to an object, such as model.forward or tensor.shape. # begins a comment.
# An object packages data and behavior. A class is a recipe for making objects.
# self means the particular object whose method is currently running.
# Lists use [a, b]; dictionaries use {key: value}; tuples often use (a, b).
# Python counts from zero. items[0] is the first item; items[-1] is the last.
# items[a:b] includes a but stops BEFORE b. An omitted boundary means the end
# or beginning. A for loop repeats a body; an if runs it only when true.
#
# THE NUMBERS AND SHAPES USED THROUGHOUT THIS FILE
# A tensor is an organized collection of numbers: a vector is a row, a matrix
# is a table, and a higher-dimensional tensor is a stack of tables. Its shape
# lists the size of each axis. B (or b) = batch size, T (or t) = sequence length,
# C = features per character, V = vocabulary size, H = attention heads,
# D = features per head = C/H. A batch processes several names together.
# Integer token IDs have shape (B,T). Embeddings have shape (B,T,C).
# Logits have shape (B,T,V): one score per possible NEXT character at each
# position. Logits are unrestricted real numbers, NOT probabilities.
# An embedding is a learned lookup table: ID 3 selects row 3, a vector of
# features. The ID itself has no numerical meaning like 'three times ID 1'.
#
# CORE MATH YOU WILL SEE REPEATED
# A linear layer computes y = W x + b: multiply inputs by learned weights,
# add them, then add a learned bias. PyTorch applies this to the last axis.
# An activation bends this calculation so stacked layers can learn nonlinear
# patterns. Without activations, stacked affine layers collapse to one.
# * usually multiplies corresponding tensor entries; @ multiplies matrices.
# For matrices, (A @ B)[i,j] = sum over k of A[i,k]*B[k,j].
# softmax turns scores z into probabilities p[i] = exp(z[i])/sum(exp(z[j])).
# These probabilities are positive and sum to 1 along the selected axis.
# Cross-entropy for a correct character c is -log(p[c]), using natural log.
# If p[c]=0.5, loss is about 0.693; if p[c]=0.1, loss is about 2.303.
# Lower loss means the correct answers received more probability on average.
# Backpropagation uses the chain rule to calculate each parameter's gradient:
# how a tiny change in that parameter would change the loss. An optimizer
# uses those gradients to update parameters toward smaller training loss.
#
# HOW TO FOLLOW THIS FILE
# 1. ModelConfig holds settings; the next classes define alternative models.
# 2. generate/evaluate use a model without updating its parameters.
# 3. CharDataset turns words into next-character exercises.
# 4. The __main__ section selects a model and runs the training loop.
# All six model choices share the interface model(inputs, optional_targets),
# which returns (logits, loss). Read CharDataset first if you want a concrete
# example of the input and target before studying the models.

# Import a standard Python module for filesystem operations, such as making
# folders and joining path components. import makes its name available here.
import os
# Import interpreter utilities; sys.exit() will end sample-only execution.
import sys
# Import clock utilities to measure how many seconds a training step takes.
import time
# Import ordinary-number math, including square root and the constant pi.
import math
# Import the command-line parser: it reads options typed after the filename.
import argparse
# from ... import ... brings one named item into this file directly.
# dataclass is a decorator that generates useful methods for settings objects.
from dataclasses import dataclass
# List is a type-annotation helper. This particular script never uses it.
from typing import List

# Import PyTorch, which provides tensors, automatic gradients, and optimizers.
# Unlike the imports above, PyTorch is a separately installed dependency.
import torch
# as nn gives torch.nn a short local name. It contains trainable layer classes.
import torch.nn as nn
# F names functions such as softmax and cross_entropy. Functions operate on
# provided tensors; layer objects such as nn.Linear also own parameters.
from torch.nn import functional as F
# Dataset is the parent class for our word-to-training-example adapter.
from torch.utils.data import Dataset
# DataLoader groups dataset examples into batches and optionally uses workers.
from torch.utils.data.dataloader import DataLoader
# SummaryWriter records numerical logs for TensorBoard charts. Importing it
# requires the TensorBoard dependency even before training starts.
from torch.utils.tensorboard import SummaryWriter


# A decorator starts with @ and modifies the definition directly below it.
# Here it creates the settings initializer rather than a neural-network layer.
@dataclass
# SETTINGS SHARED BY THE MODELS. A dataclass automatically creates an
# initializer, so ModelConfig(n_embd=64, ...) stores these named values.
# The : int annotations describe intended types; Python does not enforce them.
# None means 'not supplied yet'; dataset sizes replace those placeholders.
class ModelConfig:
    # Maximum input length, measured in character tokens. : int is a type hint;
    # = None is the initial placeholder until the dataset determines the length.
    block_size: int = None
    # Number of allowed output token IDs, including special ID 0.
    # With 26 letters this is 27; valid IDs run from 0 through 26.
    vocab_size: int = None

    # Number of Transformer blocks. Other model choices do not all use this field.
    n_layer: int = 4
    # Width of each character embedding: 64 learned feature numbers per token.
    n_embd: int = 64
    # Additional hidden width used by the MLP, recurrent models, and BoW MLP.
    n_embd2: int = 64
    # Number of attention heads. With width 64 and four heads, each head has 16 features.
    n_head: int = 4


# GELU ACTIVATION. nn.Module is PyTorch's base class for neural components.
# Inheriting from it gives parameter tracking, device movement, and model(x)
# calling behavior. This class implements a smooth nonlinear transformation.
class NewGELU(nn.Module):


    # Define the forward calculation for this component. Calling component(x)
    # uses this method through nn.Module. x is a tensor; output shapes are
    # explained at the individual operations below.
    def forward(self, x):
        # Apply the common tanh approximation to GELU, element by element:
        # GELU(x) ~= 0.5*x*[1+tanh(sqrt(2/pi)*(x+0.044715*x^3))].
        # The ideal GELU is x*Phi(x), where Phi is a standard normal cumulative
        # probability. Large negative inputs are suppressed; positive ones mostly
        # pass through. pow(x,3) cubes x; tanh smoothly stays between -1 and 1.
        # The output shape matches x, and PyTorch can differentiate this expression.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# CAUSAL MULTI-HEAD ATTENTION: each position can gather information from
# itself and earlier positions. It must not read future target characters.
# Queries ask what information is useful; keys describe what each position
# offers; values contain the information to mix. Several heads learn different
# mixtures in parallel. The labels are analogies for learned linear transforms.
class CausalSelfAttention(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Check that the embedding width splits evenly across heads. % is remainder;
        # 64 % 4 is 0. assert raises an error if this expression is false.
        assert config.n_embd % config.n_head == 0

        # Create one learned affine transform C -> 3C. Its output holds query, key,
        # and value vectors side by side. Computing them together is efficient;
        # they still have distinct learned weights. Linear includes a bias by default.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Create a learned C -> C projection to mix the concatenated head outputs.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Build a square table of ones, then tril keeps only the lower triangle.
        # For length 3 its rows are [1,0,0], [1,1,0], [1,1,1]. Row j lists the
        # positions j may read. register_buffer stores this nontrainable mask with
        # the module: it moves with model.to(...) and appears in state_dict.
        # The name bias here means a mask buffer, not a Linear layer bias.
        # The continued .view below adds leading axes for batch/head broadcasting.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # Store the head count on this attention object for use during forward.
        self.n_head = config.n_head
        # Store the full feature width C, also used to split the Q/K/V output.
        self.n_embd = config.n_embd

    # Define the forward calculation for this component. Calling component(x)
    # uses this method through nn.Module. x is a tensor; output shapes are
    # explained at the individual operations below.
    def forward(self, x):
        # size() returns the tensor shape. Unpack its three sizes into B, T, C.
        # The input is (batch, time, features), e.g. (32, 10, 64).
        B, T, C = x.size()


        # Project x from (B,T,C) to (B,T,3C), then split axis 2 into three C-wide
        # pieces. Tuple unpacking names those pieces q, k, v. Axis 2 is the third
        # axis because Python counts axes from zero.
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape (B,T,C) into (B,T,H,D), then swap time and head axes to get
        # (B,H,T,D). // is integer division, so D=C//H. view reorganizes the
        # existing entries; transpose changes axis order. Each head now has its own
        # D-dimensional vectors at every time step.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Reshape (B,T,C) into (B,T,H,D), then swap time and head axes to get
        # (B,H,T,D). // is integer division, so D=C//H. view reorganizes the
        # existing entries; transpose changes axis order. Each head now has its own
        # D-dimensional vectors at every time step.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Reshape (B,T,C) into (B,T,H,D), then swap time and head axes to get
        # (B,H,T,D). // is integer division, so D=C//H. view reorganizes the
        # existing entries; transpose changes axis order. Each head now has its own
        # D-dimensional vectors at every time step.
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        # Compute a similarity score for every query/key pair within each head.
        # transpose(-2,-1) swaps the last two axes: (B,H,T,D) -> (B,H,D,T).
        # Matrix multiplication yields (B,H,T,T): row j scores earlier/all positions
        # as information sources for position j. A dot product is sum(q[d]*k[d]).
        # Divide by sqrt(D) to control score scale as head width grows, helping
        # softmax avoid becoming extremely peaked merely because D is larger.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Slice the prebuilt mask down to the actual length T. Colons keep all
        # entries on an axis; :T keeps indices 0 through T-1. Replace forbidden
        # future scores with negative infinity. exp(-infinity)=0, so their later
        # softmax probabilities become zero. The mask broadcasts over batches/heads.
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Normalize across the last axis: the source positions being mixed.
        # Each row now sums to 1. Masked future positions have weight zero.
        # For the BoW mixer, all allowed scores were equal, so each allowed weight
        # is 1/(j+1) in row j; attention can instead learn unequal weights.
        att = F.softmax(att, dim=-1)
        # Take the weighted sum of value vectors for each query position.
        # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D). Each output feature is
        # sum over source positions of attention_weight * source_value_feature.
        y = att @ v
        # Swap back to (B,T,H,D), then join the H heads into C=H*D features.
        # transpose can produce a noncontiguous memory layout; contiguous() gives
        # a layout view() can safely reshape. The result is (B,T,C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)


        # Mix features from all heads through the learned output projection.
        y = self.c_proj(y)
        # Return the transformed tensor to the caller; return ends this method.
        return y

# ONE TRANSFORMER BLOCK. Attention mixes information across positions;
# a small feed-forward network then transforms each position independently.
# Residual additions preserve a direct path for information and gradients.
# Stacking blocks lets later calculations use earlier contextual features.
class Block(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # LayerNorm rescales each position across its C features. Roughly,
        # y = gamma*(x-mean)/sqrt(variance+epsilon)+beta. gamma and beta are
        # learned; epsilon prevents division by zero. This helps stabilize training.
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Create one attention module with its own learned projection weights.
        self.attn = CausalSelfAttention(config)
        # Create a separate normalization layer for the feed-forward sublayer.
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # Store named sublayers in ModuleDict, a dictionary that registers its
        # neural modules with PyTorch. dict(name=value, ...) creates named entries;
        # registration lets parameters(), to(), and state_dict() find the layers.
        self.mlp = nn.ModuleDict(dict(
            # Expand C features to 4C hidden features at each position independently.
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            # Project the expanded 4C hidden features back to C for residual addition.
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            # Construct the nonlinear GELU activation between the two affine layers.
            act     = NewGELU(),
        ))
        # Give the module dictionary a short local alias. This references the same
        # object; it does not copy its layers or create new weights.
        m = self.mlp
        # lambda defines a small unnamed function and stores it as self.mlpf.
        # Read nested calls inside out: expand with c_fc, activate, project back.
        # The layers already belong to self.mlp, so their weights remain registered.
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    # Define the forward calculation for this component. Calling component(x)
    # uses this method through nn.Module. x is a tensor; output shapes are
    # explained at the individual operations below.
    def forward(self, x):
        # Normalize x, compute attention, then add the result to the original x.
        # This residual/skip connection learns a correction instead of replacing
        # everything. Both summands have shape (B,T,C).
        x = x + self.attn(self.ln_1(x))
        # Normalize the updated features, run the feed-forward network, and add
        # another residual correction. Each position uses the same network weights.
        x = x + self.mlpf(self.ln_2(x))
        # Return the resulting tensor to whichever function called this method.
        return x

# TRANSFORMER LANGUAGE MODEL. Combine character and position embeddings,
# run several attention blocks, and score every possible next character.
# This uses the GPT-style decoder architecture, trained from scratch here.
# It does not load a pretrained GPT model or its knowledge.
class Transformer(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Remember the maximum context length on this particular model/component.
        self.block_size = config.block_size

        # Group the Transformer components in a registered dictionary of modules.
        self.transformer = nn.ModuleDict(dict(
            # Character embedding table of shape (V,C). Looking up (B,T) IDs gives
            # (B,T,C) features. The rows start randomly and are learned through training.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position embedding table: each position 0..block_size-1 has C learned
            # features, allowing the model to distinguish the first character from later ones.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # A list comprehension creates a fresh Block for each layer index.
            # range(n) supplies 0..n-1; _ means the index itself is unused. ModuleList
            # registers all the blocks, and each block has its own parameters.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Normalize the final block output before converting features to scores.
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Language-model head: map C features to V next-character scores.
        # bias=False omits an additive bias. This is a separate weight matrix;
        # this code does not tie it to the character embedding table.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        # parameters() yields trainable tensors; numel() counts numbers in each.
        # The generator supplies those counts to sum(). This subtotal excludes
        # lm_head because it visits only self.transformer, not the entire model.
        n_params = sum(p.numel() for p in self.transformer.parameters())
        # Print that subtotal in millions. 1e6 means 1,000,000; %.2f formats two
        # decimal places. Here % performs string formatting, rather than remainder.
        # The comma in (value,) makes a one-element tuple for the format operation.
        print("number of parameters: %.2fM" % (n_params/1e6,))

    # Expose the context length through a method shared by all model types.
    def get_block_size(self):
        # Report the stored maximum input length so generation can crop context.
        return self.block_size

    # idx contains integer input token IDs, shape (b,t). Optional targets
    # contain desired next-token IDs at matching positions, with -1 for ignored
    # padding. The default targets=None permits prediction without known answers.
    def forward(self, idx, targets=None):
        # Read where input IDs live: CPU memory or a supported accelerator.
        # New tensors used with idx must be created on a compatible device.
        # In the recurrent implementation this local name is not used afterward.
        device = idx.device
        # Unpack the integer input shape: b examples, t character positions each.
        b, t = idx.size()
        # Reject input longer than the configured context. An f-string inserts
        # the values in {...} into the error message if the assertion fails.
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # Create integer positions [0,1,...,t-1] on the same device as idx.
        # long means 64-bit integer, the type embedding indices expect.
        # unsqueeze(0) adds a leading axis: (t,) becomes (1,t).
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)


        # Look up character vectors: input (b,t) becomes features (b,t,C).
        tok_emb = self.transformer.wte(idx)
        # Look up position vectors: (1,t) becomes (1,t,C).
        pos_emb = self.transformer.wpe(pos)
        # Add character identity and position information. Broadcasting reuses
        # the (1,t,C) positions for every row of the (b,t,C) character batch;
        # the result stays (b,t,C). This is elementwise addition, not concatenation.
        x = tok_emb + pos_emb
        # Visit the registered Transformer blocks in order, reusing their learned
        # weights every time the model is called.
        for block in self.transformer.h:
            # Pass the current features through this block. PyTorch module calling
            # runs forward() and returns its output as the next block input.
            x = block(x)
        # Apply the final feature normalization, keeping shape (b,t,C).
        x = self.transformer.ln_f(x)
        # Convert features to raw next-token scores, shape (b,t,V).
        # No softmax is needed here: cross_entropy handles it internally for training.
        logits = self.lm_head(x)


        # Default to no loss for generation calls, where correct targets are absent.
        # None represents the absence of a value, not the numeric value zero.
        loss = None
        # Only score prediction errors when a target tensor was provided.
        # is not None tests whether the argument is present; do not use tensor
        # truth-value tests to decide whether a whole target tensor exists.
        if targets is not None:
            # Flatten batch and time: (b,t,V) -> (b*t,V), targets (b,t) -> (b*t,).
            # size(-1) gets the final-axis size V; -1 in view asks PyTorch to infer
            # the flattened dimension from the total number of entries.
            # For each nonignored target c, compute -log softmax(logits)[c], then
            # average those losses. Internally this uses stable log-softmax math;
            # we supply raw scores rather than precomputed probabilities.
            # ignore_index=-1 excludes padded target positions from both the sum
            # and the averaging count. This -1 has a DIFFERENT role from view(-1).
            # The scalar loss keeps its computation graph so backward() can trace
            # how all participating trainable numbers influenced the error.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return a two-item tuple: scores and optional scalar loss. The caller
        # can unpack it as logits, loss = model(...).
        return logits, loss


# CAUSAL BAG-OF-WORDS MIXER. Instead of learning attention preferences,
# give every allowed position equal weight. At position j, average feature
# vectors from positions 0 through j, INCLUDING j. No future positions enter.
# Despite the name, this program mixes characters, not whole words.
class CausalBoW(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()


        # Remember the maximum context length on this particular model/component.
        self.block_size = config.block_size
        # Build a square table of ones, then tril keeps only the lower triangle.
        # For length 3 its rows are [1,0,0], [1,1,0], [1,1,1]. Row j lists the
        # positions j may read. register_buffer stores this nontrainable mask with
        # the module: it moves with model.to(...) and appears in state_dict.
        # The name bias here means a mask buffer, not a Linear layer bias.
        # The continued .view below adds leading axes for batch/head broadcasting.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    # Define the forward calculation for this component. Calling component(x)
    # uses this method through nn.Module. x is a tensor; output shapes are
    # explained at the individual operations below.
    def forward(self, x):
        # size() returns the tensor shape. Unpack its three sizes into B, T, C.
        # The input is (batch, time, features), e.g. (32, 10, 64).
        B, T, C = x.size()


        # Start every context-mixing score at zero, on the input device.
        # There are T candidate source positions for each of T output positions.
        att = torch.zeros((B, T, T), device=x.device)
        # Hide future positions with negative infinity, using the length-T portion
        # of the triangular mask. Current and past positions retain score zero.
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        # Normalize across the last axis: the source positions being mixed.
        # Each row now sums to 1. Masked future positions have weight zero.
        # For the BoW mixer, all allowed scores were equal, so each allowed weight
        # is 1/(j+1) in row j; attention can instead learn unequal weights.
        att = F.softmax(att, dim=-1)
        # Average allowed input vectors: (B,T,T) @ (B,T,C) -> (B,T,C).
        # For example, row 2 becomes (x[0]+x[1]+x[2])/3 within each example.
        y = att @ x

        # Return the transformed tensor to the caller; return ends this method.
        return y

# BAG-OF-WORDS BLOCK. First add the average available context to each
# position, then apply a nonlinear feature transformation with a residual.
class BoWBlock(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()


        # Construct the prefix-average context mixer; its mask is nontrainable.
        self.cbow = CausalBoW(config)

        # Store named sublayers in ModuleDict, a dictionary that registers its
        # neural modules with PyTorch. dict(name=value, ...) creates named entries;
        # registration lets parameters(), to(), and state_dict() find the layers.
        self.mlp = nn.ModuleDict(dict(
            # Map the C input features to the configured second hidden width.
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            # Map the hidden features back to C so residual shapes agree.
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        # Give the module dictionary a short local alias. This references the same
        # object; it does not copy its layers or create new weights.
        m = self.mlp
        # Store a small forward function: first affine layer, tanh activation,
        # second affine layer. tanh(a)=(exp(a)-exp(-a))/(exp(a)+exp(-a));
        # it smoothly compresses each number into the interval (-1,1).
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    # Define the forward calculation for this component. Calling component(x)
    # uses this method through nn.Module. x is a tensor; output shapes are
    # explained at the individual operations below.
    def forward(self, x):
        # Add the prefix average to the original position features. Current and
        # past positions contribute; there is no access to future characters.
        x = x + self.cbow(x)
        # Add the nonlinear MLP correction, leaving the feature shape unchanged.
        x = x + self.mlpf(x)
        # Return the resulting tensor to whichever function called this method.
        return x

# BAG-OF-WORDS LANGUAGE MODEL. Learned position embeddings provide some
# position information before the prefix average. The mixing weights are
# uniform rather than the learned, content-dependent weights in attention.
class BoW(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Remember the maximum context length on this particular model/component.
        self.block_size = config.block_size
        # Store V, the number of possible output tokens, including STOP/START ID 0.
        self.vocab_size = config.vocab_size

        # Create the trainable character lookup table, with V rows and C columns.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Create a trainable C-feature vector for every permitted position index.
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Create the prefix-averaging and feed-forward block used by this model.
        self.context_block = BoWBlock(config)

        # Create the final C -> V affine layer, including a learned bias by default.
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    # Expose the context length through a method shared by all model types.
    def get_block_size(self):
        # Report the stored maximum input length so generation can crop context.
        return self.block_size

    # idx contains integer input token IDs, shape (b,t). Optional targets
    # contain desired next-token IDs at matching positions, with -1 for ignored
    # padding. The default targets=None permits prediction without known answers.
    def forward(self, idx, targets=None):

        # Read where input IDs live: CPU memory or a supported accelerator.
        # New tensors used with idx must be created on a compatible device.
        # In the recurrent implementation this local name is not used afterward.
        device = idx.device
        # Unpack the integer input shape: b examples, t character positions each.
        b, t = idx.size()
        # Reject input longer than the configured context. An f-string inserts
        # the values in {...} into the error message if the assertion fails.
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # Create integer positions [0,1,...,t-1] on the same device as idx.
        # long means 64-bit integer, the type embedding indices expect.
        # unsqueeze(0) adds a leading axis: (t,) becomes (1,t).
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)


        # Look up the feature vector for every integer in idx: (b,t) -> (b,t,C).
        tok_emb = self.wte(idx)
        # Look up position features, shape (1,t,C), shared across examples.
        pos_emb = self.wpe(pos)

        # Add character identity and position information. Broadcasting reuses
        # the (1,t,C) positions for every row of the (b,t,C) character batch;
        # the result stays (b,t,C). This is elementwise addition, not concatenation.
        x = tok_emb + pos_emb

        # Mix each position with its available prefix, then transform its features.
        x = self.context_block(x)

        # Convert features to raw next-token scores, shape (b,t,V).
        # No softmax is needed here: cross_entropy handles it internally for training.
        logits = self.lm_head(x)


        # Default to no loss for generation calls, where correct targets are absent.
        # None represents the absence of a value, not the numeric value zero.
        loss = None
        # Only score prediction errors when a target tensor was provided.
        # is not None tests whether the argument is present; do not use tensor
        # truth-value tests to decide whether a whole target tensor exists.
        if targets is not None:
            # Flatten batch and time: (b,t,V) -> (b*t,V), targets (b,t) -> (b*t,).
            # size(-1) gets the final-axis size V; -1 in view asks PyTorch to infer
            # the flattened dimension from the total number of entries.
            # For each nonignored target c, compute -log softmax(logits)[c], then
            # average those losses. Internally this uses stable log-softmax math;
            # we supply raw scores rather than precomputed probabilities.
            # ignore_index=-1 excludes padded target positions from both the sum
            # and the averaging count. This -1 has a DIFFERENT role from view(-1).
            # The scalar loss keeps its computation graph so backward() can trace
            # how all participating trainable numbers influenced the error.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return a two-item tuple: scores and optional scalar loss. The caller
        # can unpack it as logits, loss = model(...).
        return logits, loss


# ONE RECURRENT NEURAL NETWORK STEP. A hidden state is a vector serving
# as a running numerical summary of the prefix. Combine the current character
# vector with the previous summary, then make the next summary. The SAME
# cell weights are reused at every character position.
class RNNCell(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Create an affine transform from the joined character vector and previous
        # hidden state to a new hidden vector. If widths are C and S, this is
        # (C+S) -> S, equivalent to W_x*x_t + W_h*h_previous + bias.
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    # Calculate ONE sequence step from current embeddings xt, shape (b,C),
    # and the previous hidden summary hprev, shape (b,S).
    def forward(self, xt, hprev):
        # Join current character features and previous hidden features side by
        # side: (b,C) plus (b,S) -> (b,C+S). cat means concatenate; it preserves
        # all entries rather than adding matching entries together.
        xh = torch.cat([xt, hprev], dim=1)
        # Compute h_t = tanh(W*[x_t,h_previous]+bias). The tanh acts on each
        # hidden feature independently. Output shape is (b,S).
        ht = F.tanh(self.xh_to_h(xh))
        # Return the new hidden state; the outer RNN loop will reuse it next step.
        return ht

# GATED RECURRENT UNIT STEP. A plain RNN replaces its summary each step;
# a GRU learns gates that control how old and new information are combined.
# A gate is a vector of values between 0 and 1, one per hidden feature.
# Here r controls the old information used to build a candidate; z controls
# how much that candidate replaces the previous state.
class GRUCell(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()

        # Create the learned affine transform for the GRU replacement gate.
        # Its input is C+S features and its output is S hidden features.
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        # Create the learned affine transform for the GRU reset gate.
        # Its input is C+S features and its output is S hidden features.
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        # Create the learned affine transform for the GRU candidate state.
        # Its input is C+S features and its output is S hidden features.
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    # Calculate ONE sequence step from current embeddings xt, shape (b,C),
    # and the previous hidden summary hprev, shape (b,S).
    def forward(self, xt, hprev):

        # Join current character features and previous hidden features side by
        # side: (b,C) plus (b,S) -> (b,C+S). cat means concatenate; it preserves
        # all entries rather than adding matching entries together.
        xh = torch.cat([xt, hprev], dim=1)
        # Compute reset gates r = sigmoid(W_r*[x_t,h_previous]+b_r).
        # sigmoid(a)=1/(1+exp(-a)), so each gate lies between 0 and 1.
        r = F.sigmoid(self.xh_to_r(xh))
        # Multiply feature by feature. r near 0 hides an old feature from the
        # candidate calculation; r near 1 lets that feature through. This does not
        # modify hprev in place, so the original remains available for blending.
        hprev_reset = r * hprev

        # Join the current character with the reset-filtered old hidden state.
        xhr = torch.cat([xt, hprev_reset], dim=1)
        # Build the candidate new summary, with entries between -1 and 1.
        # It is a proposal, not yet the final hidden state.
        hbar = F.tanh(self.xh_to_hbar(xhr))

        # Compute the replacement gates from the original joined input/state.
        # In THIS convention, z near 1 favors the candidate; near 0 keeps the old
        # state. Some GRU descriptions use the opposite naming convention.
        z = F.sigmoid(self.xh_to_z(xh))

        # Blend old and candidate states per feature: h_t=(1-z)*h_previous+z*hbar.
        # If old=0.2, candidate=0.8, z=0.25, the result is 0.75*0.2+0.25*0.8=0.35.
        # The network learns these gates through gradients, rather than explicit rules.
        ht = (1 - z) * hprev + z * hbar
        # Return the new hidden state; the outer RNN loop will reuse it next step.
        return ht

# RECURRENT LANGUAGE MODEL. Embed the input characters, visit them in
# sequence, and use each hidden state to predict the next character. cell_type
# selects either the plain RNN cell or the GRU cell defined above.
# Unlike attention, the recurrent state updates must follow time order.
class RNN(nn.Module):

    # Initialize a recurrent model; cell_type selects plain or gated updates.
    def __init__(self, config, cell_type):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Remember the maximum context length on this particular model/component.
        self.block_size = config.block_size
        # Store V, the number of possible output tokens, including STOP/START ID 0.
        self.vocab_size = config.vocab_size
        # Create a LEARNED initial hidden state, shape (1,S), initialized to zeros.
        # nn.Parameter registers these numbers for optimization, unlike a buffer.
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))
        # Create the trainable character lookup table, with V rows and C columns.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Choose the ordinary recurrent update when cell_type equals the string rnn.
        if cell_type == 'rnn':
            # Construct one recurrent cell; all time steps share this same instance.
            self.cell = RNNCell(config)
        # elif means else-if: try this branch only if the previous condition failed.
        elif cell_type == 'gru':
            # Use a gated recurrent cell instead, still shared across time steps.
            self.cell = GRUCell(config)
        # Map each S-wide hidden summary to V scores for the next character.
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    # Expose the context length through a method shared by all model types.
    def get_block_size(self):
        # Report the stored maximum input length so generation can crop context.
        return self.block_size

    # idx contains integer input token IDs, shape (b,t). Optional targets
    # contain desired next-token IDs at matching positions, with -1 for ignored
    # padding. The default targets=None permits prediction without known answers.
    def forward(self, idx, targets=None):
        # Read where input IDs live: CPU memory or a supported accelerator.
        # New tensors used with idx must be created on a compatible device.
        # In the recurrent implementation this local name is not used afterward.
        device = idx.device
        # Unpack the integer input shape: b examples, t character positions each.
        b, t = idx.size()


        # Embed all input IDs at once, giving (b,t,C). The recurrent state
        # updates remain sequential, but these independent lookups can be batched.
        emb = self.wte(idx)


        # Reuse the learned initial state for all b names: (1,S) -> (b,S).
        # expand creates a broadcasted view rather than independent parameters.
        # Here -1 means keep the existing size S. Gradients from examples accumulate
        # into the same learned starting state. Each forward call starts afresh.
        hprev = self.start.expand((b, -1))
        # Start an empty Python list to collect the hidden output at each position.
        hiddens = []
        # Visit character positions 0 through t-1 in order. Later states depend
        # on earlier states, so this loop carries the sequence memory forward.
        for i in range(t):
            # Take position i from every batch row, keeping all feature columns:
            # (b,t,C) -> (b,C). A plain integer index removes the selected axis.
            xt = emb[:, i, :]
            # Call the chosen cell to calculate the next hidden state, shape (b,S).
            ht = self.cell(xt, hprev)
            # Use this new state as the previous state for the next character.
            # The computation graph still links all steps for backpropagation through time.
            hprev = ht
            # append stores this time step tensor at the end of the Python list.
            hiddens.append(ht)


        # stack creates a new time axis: t tensors of shape (b,S) -> (b,t,S).
        # Unlike cat, stack introduces an axis rather than extending an existing one.
        hidden = torch.stack(hiddens, 1)
        # Convert each time step summary to next-character scores: (b,t,S) -> (b,t,V).
        logits = self.lm_head(hidden)


        # Default to no loss for generation calls, where correct targets are absent.
        # None represents the absence of a value, not the numeric value zero.
        loss = None
        # Only score prediction errors when a target tensor was provided.
        # is not None tests whether the argument is present; do not use tensor
        # truth-value tests to decide whether a whole target tensor exists.
        if targets is not None:
            # Flatten batch and time: (b,t,V) -> (b*t,V), targets (b,t) -> (b*t,).
            # size(-1) gets the final-axis size V; -1 in view asks PyTorch to infer
            # the flattened dimension from the total number of entries.
            # For each nonignored target c, compute -log softmax(logits)[c], then
            # average those losses. Internally this uses stable log-softmax math;
            # we supply raw scores rather than precomputed probabilities.
            # ignore_index=-1 excludes padded target positions from both the sum
            # and the averaging count. This -1 has a DIFFERENT role from view(-1).
            # The scalar loss keeps its computation graph so backward() can trace
            # how all participating trainable numbers influenced the error.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return a two-item tuple: scores and optional scalar loss. The caller
        # can unpack it as logits, loss = model(...).
        return logits, loss


# MULTILAYER PERCEPTRON LANGUAGE MODEL. At every position, collect a fixed
# window ending at that position, join its embeddings, and pass them through
# a small network. Each offset gets its own section of the joined vector,
# so the network can distinguish order. A special BLANK fills missing history.
class MLP(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Remember the maximum context length on this particular model/component.
        self.block_size = config.block_size
        # Store V, the number of possible output tokens, including STOP/START ID 0.
        self.vocab_size = config.vocab_size
        # Reserve V+1 embedding rows: ordinary IDs 0..V-1 plus BLANK ID V.
        # BLANK represents missing history before the sequence starts. It differs
        # from START/STOP ID 0 and is never a valid predicted output character.
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)


        # Sequential runs layers in the listed order, automatically passing each
        # output to the next. These layer objects and their weights are registered.
        self.mlp = nn.Sequential(
            # The concatenated window contains block_size*C features. Map them to
            # the hidden width S. This layer learns different weights for each offset.
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            # Insert a nonlinear tanh activation so the two linear layers can express
            # more than a single affine transformation.
            nn.Tanh(),
            # Map hidden features to V next-token scores at every input position.
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    # Expose the context length through a method shared by all model types.
    def get_block_size(self):
        # Report the stored maximum input length so generation can crop context.
        return self.block_size

    # idx contains integer input token IDs, shape (b,t). Optional targets
    # contain desired next-token IDs at matching positions, with -1 for ignored
    # padding. The default targets=None permits prediction without known answers.
    def forward(self, idx, targets=None):


        # Collect one embedding tensor for each history offset in the fixed window.
        embs = []
        # Repeat once per context offset: current input, one step earlier, and
        # so on. k is the loop index, although the body uses repeated shifts instead.
        for k in range(self.block_size):
            # Look up the feature vector for every integer in idx: (b,t) -> (b,t,C).
            tok_emb = self.wte(idx)
            # Shift token IDs right by one position along time axis 1. roll wraps the
            # last entry around to the first; the next line removes that wrapped entry.
            # This produces a new tensor, so the following assignment does not overwrite
            # the caller input. Repeating shifts exposes increasingly old history.
            idx = torch.roll(idx, 1, 1)
            # Overwrite the first position of each row with BLANK ID V, preventing
            # wrapped future characters from leaking into the left boundary.
            idx[:, 0] = self.vocab_size
            # Save embeddings BEFORE the shift as this offset contribution. At position
            # j, these vectors represent input j, j-1, ...; missing positions use BLANK.
            embs.append(tok_emb)


        # Join embeddings along the final feature axis, keeping batch/time intact:
        # (b,t,C) repeated block_size times -> (b,t,C*block_size).
        x = torch.cat(embs, -1)
        # Apply the fixed-window neural network to predict each next token.
        logits = self.mlp(x)


        # Default to no loss for generation calls, where correct targets are absent.
        # None represents the absence of a value, not the numeric value zero.
        loss = None
        # Only score prediction errors when a target tensor was provided.
        # is not None tests whether the argument is present; do not use tensor
        # truth-value tests to decide whether a whole target tensor exists.
        if targets is not None:
            # Flatten batch and time: (b,t,V) -> (b*t,V), targets (b,t) -> (b*t,).
            # size(-1) gets the final-axis size V; -1 in view asks PyTorch to infer
            # the flattened dimension from the total number of entries.
            # For each nonignored target c, compute -log softmax(logits)[c], then
            # average those losses. Internally this uses stable log-softmax math;
            # we supply raw scores rather than precomputed probabilities.
            # ignore_index=-1 excludes padded target positions from both the sum
            # and the averaging count. This -1 has a DIFFERENT role from view(-1).
            # The scalar loss keeps its computation graph so backward() can trace
            # how all participating trainable numbers influenced the error.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return a two-item tuple: scores and optional scalar loss. The caller
        # can unpack it as logits, loss = model(...).
        return logits, loss


# BIGRAM MODEL: the simplest alternative. Only the current character
# helps predict the next one. A trainable V-by-V table stores next-character
# scores: row = current token, column = next token. There is no hidden layer.
# For example, the row for 'q' can learn a large score for 'u'.
class Bigram(nn.Module):


    # Construct this component from the settings object. __init__ runs when
    # you create an instance, e.g. Transformer(config). Python supplies self;
    # config is supplied by the caller. Store persistent values with self.name.
    def __init__(self, config):
        # Initialize the parent nn.Module machinery before registering layers or
        # parameters. super() refers to the inherited implementation; self is implicit.
        super().__init__()
        # Use n as a shorter local name for vocabulary size V.
        n = config.vocab_size
        # Create V*V trainable scores initialized to zero. With equal scores,
        # softmax initially gives a uniform 1/V probability to every next token.
        self.logits = nn.Parameter(torch.zeros((n, n)))

    # Expose the context length through a method shared by all model types.
    def get_block_size(self):
        # The bigram model needs only one previous token during generation.
        return 1

    # idx contains integer input token IDs, shape (b,t). Optional targets
    # contain desired next-token IDs at matching positions, with -1 for ignored
    # padding. The default targets=None permits prediction without known answers.
    def forward(self, idx, targets=None):


        # Use input token IDs to select rows from the score table. Indexing a
        # (V,V) table with (b,t) IDs gives (b,t,V) scores. Backpropagation updates
        # the selected rows; there is no embedding or matrix multiply in this model.
        logits = self.logits[idx]


        # Default to no loss for generation calls, where correct targets are absent.
        # None represents the absence of a value, not the numeric value zero.
        loss = None
        # Only score prediction errors when a target tensor was provided.
        # is not None tests whether the argument is present; do not use tensor
        # truth-value tests to decide whether a whole target tensor exists.
        if targets is not None:
            # Flatten batch and time: (b,t,V) -> (b*t,V), targets (b,t) -> (b*t,).
            # size(-1) gets the final-axis size V; -1 in view asks PyTorch to infer
            # the flattened dimension from the total number of entries.
            # For each nonignored target c, compute -log softmax(logits)[c], then
            # average those losses. Internally this uses stable log-softmax math;
            # we supply raw scores rather than precomputed probabilities.
            # ignore_index=-1 excludes padded target positions from both the sum
            # and the averaging count. This -1 has a DIFFERENT role from view(-1).
            # The scalar loss keeps its computation graph so backward() can trace
            # how all participating trainable numbers influenced the error.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return a two-item tuple: scores and optional scalar loss. The caller
        # can unpack it as logits, loss = model(...).
        return logits, loss


# Run the decorated generation function without recording gradient graphs.
# Sampling needs predictions, not derivatives; this reduces memory use.
# This decorator does not automatically switch the model to eval() mode.
@torch.no_grad()
# Complete a batch of prefixes. idx is (b,t); the return adds new token
# columns. Defaults use temperature 1, greedy selection, and no top-k filter.
# Generation repeatedly predicts one step, appends it, then predicts again.
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):


    # Ask this model how much context it accepts; bigram returns 1.
    block_size = model.get_block_size()
    # Add exactly max_new_tokens columns. _ is an unused loop variable.
    # This loop does not stop individually at STOP; display logic crops later.
    for _ in range(max_new_tokens):

        # This is a conditional expression: use all IDs if they fit; otherwise
        # use only the newest block_size columns. Negative slice starts count
        # from the end. The full generated history remains in idx.
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Get predictions for the cropped context. No targets means the returned
        # loss is None; _ receives that unused second result.
        logits, _ = model(idx_cond)

        # Select the final time position: (b,t,V) -> (b,V), because that predicts
        # the character to append. Divide scores by a positive temperature.
        # Below 1 sharpens preferences; above 1 flattens them. Temperature must be
        # positive; this function does not validate it. Default 1 leaves scores alone.
        logits = logits[:, -1, :] / temperature

        # Optionally restrict sampling to candidates with the largest scores.
        # None means no restriction; a supplied k should lie between 1 and V.
        if top_k is not None:
            # Get the k largest scores per row, sorted largest first. The second
            # returned tensor contains their indices and is unused here.
            v, _ = torch.topk(logits, top_k)
            # Use the kth-largest score as a cutoff; replace smaller scores with
            # negative infinity so they get zero probability. [:,[-1]] retains a
            # length-1 axis, giving (b,1) for broadcasting against (b,V). Equal scores
            # at the threshold all survive, so ties can leave more than k candidates.
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Turn each row of scores into probabilities across V possible tokens.
        probs = F.softmax(logits, dim=-1)

        # Choose whether to draw randomly according to probabilities or pick a maximum.
        if do_sample:
            # Draw one token per row, shape (b,1). A character with probability 0.7
            # is selected about 70% of repeated draws; less likely choices allow variety.
            idx_next = torch.multinomial(probs, num_samples=1)
        # Fallback branch: run this only when the preceding if/elif conditions fail.
        else:
            # Greedy alternative: take the index of a highest-probability character.
            # Discard the returned probability values; keep indices shaped (b,1).
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # Append one new time column to every sequence. That chosen character
        # becomes input for the next iteration: this is autoregressive generation.
        idx = torch.cat((idx, idx_next), dim=1)

    # Return the initial prefix plus all newly generated token columns.
    return idx

# Generate and display num words (default 10). This helper uses the global
# model, args, train_dataset, and test_dataset initialized by the entry point.
def print_samples(num=10):

    # Start num sequences with one integer zero each: shape (num,1).
    # Zero means START in this position. to(...) moves IDs to the model device.
    # This helper reads global args, model, and datasets created in __main__.
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    # Translate command-line sentinel -1 into Python None for unrestricted sampling.
    top_k = args.top_k if args.top_k != -1 else None
    # Generate at most max_word_length characters; the existing START column
    # already uses one of the max_word_length+1 available columns. Display
    # still works if no STOP appears, by keeping the whole generated suffix.
    steps = train_dataset.get_output_length() - 1
    # Generate random continuations, then move IDs to CPU for Python decoding.
    # The default temperature is 1.0 because this call does not override it.
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    # Create three distinct lists to group exact text matches after sampling.
    train_samples, test_samples, new_samples = [], [], []
    # Loop over each generated batch row; size(0) is the number of sampled names.
    for i in range(X_samp.size(0)):

        # Select row i and skip its initial START token. tolist() converts tensor
        # entries to ordinary Python integers for membership checks and decoding.
        row = X_samp[i, 1:].tolist()

        # If a STOP zero occurs, find its first position; otherwise use the row
        # length. Checking membership first avoids index() failing when zero is absent.
        crop_index = row.index(0) if 0 in row else len(row)
        # Keep characters before STOP, excluding STOP itself and everything after.
        row = row[:crop_index]
        # Convert the retained positive token IDs back into a Python string.
        word_samp = train_dataset.decode(row)

        # Check whether the sampled text exactly matches a training example.
        if train_dataset.contains(word_samp):
            # Add this matching string to the training-match list. Duplicates are kept.
            train_samples.append(word_samp)
        # Otherwise check for an exact match in the held-out test words.
        elif test_dataset.contains(word_samp):
            # Add the string to the test-match list.
            test_samples.append(word_samp)
        # Fallback branch: run this only when the preceding if/elif conditions fail.
        else:
            # Keep strings absent from both lists as new samples. New means an exact
            # text match was absent, not that the result is necessarily a realistic name.
            new_samples.append(word_samp)
    # String multiplication repeats a dash 80 times to print a visual separator.
    print('-'*80)
    # Loop over three (list, description) pairs. Unpack each pair into lst
    # and desc so the same printing code handles every group.
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        # An f-string inserts the group length and description into a heading.
        print(f"{len(lst)} samples that are {desc}:")
        # Visit every sampled word in the current group.
        for word in lst:
            # Print one word on its own line.
            print(word)
    # String multiplication repeats a dash 80 times to print a visual separator.
    print('-'*80)

# Disable gradient recording plus additional autograd bookkeeping during
# evaluation. This is for measuring predictions, not for training.
@torch.inference_mode()
# Estimate model loss on a dataset without training. The batch size
# defaults to 50; max_batches optionally limits how far the loader is read.
def evaluate(model, dataset, batch_size=50, max_batches=None):
    # Switch model behavior to evaluation mode. This affects layers such as
    # dropout in other models; the models here have no dropout. eval() by
    # itself does not disable gradients, hence the decorator above.
    model.eval()
    # Build finite batches from this dataset, in shuffled order. Zero workers
    # means loading happens in this process; the CLI worker option is not used
    # by this evaluation helper. A final batch can be smaller than batch_size.
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    # Collect one ordinary scalar loss per evaluated batch.
    losses = []
    # enumerate gives (index, value) pairs. i starts at zero; batch contains
    # stacked input/target tensors from dataset examples.
    for i, batch in enumerate(loader):
        # List comprehension: move each tensor t in the batch to the chosen
        # compute device. It returns a new list holding the moved input and target.
        batch = [t.to(args.device) for t in batch]
        # Unpack the batch: X is input IDs and Y is the desired next-character IDs.
        # Both have shape (batch_size, sequence_length), aside from a shorter final batch.
        X, Y = batch
        # Calling an nn.Module invokes its forward method through PyTorch.
        # Supply input AND targets to obtain predictions and cross-entropy loss.
        # During training a gradient graph is built; evaluation decorators disable it.
        logits, loss = model(X, Y)
        # item() extracts the scalar tensor value as a Python number. Keep it
        # for reporting, without retaining a computation graph.
        losses.append(loss.item())
        # Stop after processing the batch whose zero-based index reaches the
        # limit. As written, max_batches=10 permits indices 0..10: ELEVEN batches
        # if that many exist. and requires both conditions to be true.
        if max_batches is not None and i >= max_batches:
            # Exit the nearest enclosing loop immediately, continuing after that loop.
            break
    # Average the collected batch losses and return a Python number.
    # This weights each batch equally, not each valid target equally, so it
    # is an approximate dataset metric when valid-token counts differ.
    mean_loss = torch.tensor(losses).mean().item()
    # Put the model back into training mode before returning to its caller.
    # This helper always enables training afterward rather than restoring a saved mode.
    model.train()
    # Return the estimated error; smaller is better for this prediction task.
    return mean_loss


# TURN WORDS INTO SUPERVISED EXERCISES. Inheriting Dataset gives PyTorch
# a standard interface: len(dataset) counts examples and dataset[i] retrieves
# one (input, target) pair. Each word is one example containing many next-token
# predictions. Token 0 means START at the beginning and STOP as a target.
# Real characters start at ID 1; -1 is an ignored TARGET, never an input ID.
class CharDataset(Dataset):

    # Store the words and build shared reversible character-ID dictionaries.
    def __init__(self, words, chars, max_word_length):
        # Store the list of word strings for this dataset split.
        self.words = words
        # Store the shared sorted vocabulary of actual characters.
        self.chars = chars
        # Store the longest word length, shared by both splits so tensor sizes agree.
        self.max_word_length = max_word_length
        # Dictionary comprehension creates string-to-integer mapping: e.g.
        # {a:1,b:2,...}, with actual string keys. enumerate supplies index i and
        # character ch; i+1 reserves ID 0 for START/STOP.
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        # Reverse the mapping: integer-to-string. items() yields (character, ID)
        # pairs; this comprehension swaps each pair to {ID: character}.
        self.itos = {i:s for s,i in self.stoi.items()}

    # Python special method: len(dataset) calls this automatically.
    def __len__(self):
        # len(...) counts words; Python uses this method for len(dataset).
        return len(self.words)

    # Check whether this exact text appears in this dataset split.
    def contains(self, word):
        # in checks exact membership and returns True or False. A Python list
        # search scans entries; it does not measure similarity between names.
        return word in self.words

    # Report how many token choices the output classifier must score.
    def get_vocab_size(self):
        # Count real characters plus the one shared START/STOP ID 0.
        return len(self.chars) + 1

    # Report the padded number of time positions in one training example.
    def get_output_length(self):
        # Reserve one extra time position beyond the longest word so training
        # can include START as input and STOP as a target.
        return self.max_word_length + 1

    # Translate a Python string into a tensor of real-character token IDs.
    def encode(self, word):
        # Visit each character w in the word, look up its ID, and build an
        # integer tensor. For a=1,n=14, the word ana becomes [1,14,1].
        # An unknown character would raise KeyError; vocabulary came from this data.
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        # Return the encoded 1D sequence of character IDs, without boundary tokens.
        return ix

    # Translate real-character IDs back into text. The caller handles boundaries.
    def decode(self, ix):
        # Look up each ID and join its characters with an empty separator.
        # The generator expression produces characters one at a time. This mapping
        # contains only real characters; the sampling helper removes STOP first.
        word = ''.join(self.itos[i] for i in ix)
        # Return the decoded Python string.
        return word

    # Python special method: dataset[idx] calls this. Build the input and
    # next-character target for a single word, using a fixed padded length.
    def __getitem__(self, idx):
        # Select one word by its dataset index. This idx is a single example
        # index, unlike the batched token tensor named idx in model.forward.
        word = self.words[idx]
        # Convert this word to its positive character IDs.
        ix = self.encode(word)
        # Allocate fixed-length input IDs, initially all zero. Fixed lengths let
        # DataLoader stack different-length words into one rectangular batch.
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        # Allocate target IDs, also initially zero. The zero immediately after
        # the real target characters will remain as the STOP prediction target.
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        # Copy letters into input starting at position 1; leave position 0 as START.
        # Example word ana, max length 5: X=[0,a,n,a,0,0], using letters for IDs.
        # Slice assignment writes into the existing tensor rather than creating a new one.
        x[1:1+len(ix)] = ix
        # Targets start one position earlier than input letters:
        # Y=[a,n,a,0,0,0] before ignoring padding. Pair each X position with its
        # next character: START->a, a->n, n->a, last a->STOP.
        y[:len(ix)] = ix
        # Ignore every target AFTER the one STOP target. In the example,
        # Y=[a,n,a,0,-1,-1]. Training learns to stop once, then pays no loss for
        # the remaining padding. Padding inputs remain 0; causal models prevent
        # these later padding positions from affecting earlier valid predictions.
        y[len(ix)+1:] = -1
        # Return one exercise pair. DataLoader stacks many such pairs into (X,Y).
        return x, y

# Prepare vocabulary, train/test split, and their Dataset adapters from text.
def create_datasets(input_file):


    # Open the text file for reading. with automatically closes it when the
    # indented block ends, including if an exception occurs. Relative input
    # paths are resolved from the process working directory.
    with open(input_file, 'r') as f:
        # Read the entire file into one Python string.
        data = f.read()
    # Split the string at line boundaries to get one candidate word per line.
    words = data.splitlines()
    # Build a new list with surrounding whitespace removed from each word.
    # Whitespace inside a word is not removed.
    words = [w.strip() for w in words]
    # Keep only nonempty strings: empty strings are false in a condition.
    words = [w for w in words if w]
    # Join all words, use set to keep unique characters, turn them into a
    # list, then sort. Sorting makes the character-to-ID order stable for the
    # same vocabulary. Repeated word examples themselves are NOT deduplicated.
    chars = sorted(list(set(''.join(words))))
    # Find the maximum character count. The generator supplies each length
    # to max(). The file must contain at least one nonempty word.
    max_word_length = max(len(w) for w in words)
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(f"number of examples in the dataset: {len(words)}")
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(f"max word length: {max_word_length}")
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print("vocabulary:")
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(''.join(chars))


    # Hold out 10% of word entries, rounded down, capped at 1000.
    # int truncates the positive decimal; min picks the smaller value.
    # For fewer than 10 words this becomes zero; the slices below then produce
    # an empty training split because -0 equals 0. This code assumes enough data.
    test_set_size = min(1000, int(len(words) * 0.1))
    # Generate a random permutation of word indices, then turn it into a
    # Python list. The earlier random seed controls this shuffle.
    rp = torch.randperm(len(words)).tolist()
    # Use shuffled entries except the held-out tail for training. Negative
    # slice boundaries count backward from the end of the index list.
    train_words = [words[i] for i in rp[:-test_set_size]]
    # Use the last test_set_size entries for evaluation. Training updates do
    # not use these entries. Duplicate word text can still appear in both
    # splits because the split is by entry rather than unique word.
    test_words = [words[i] for i in rp[-test_set_size:]]
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")


    # Create the training adapter with the shared vocabulary and padded length.
    train_dataset = CharDataset(train_words, chars, max_word_length)
    # Create the held-out adapter using exactly the same token IDs and shapes.
    test_dataset = CharDataset(test_words, chars, max_word_length)

    # Return the two adapters as a tuple for the caller to unpack.
    return train_dataset, test_dataset

# KEEP SUPPLYING RANDOM TRAINING BATCHES. This wrapper uses a very large
# sampler, then restarts its iterator if it eventually ends. Sampling with
# replacement means a name can appear more than once, even within a batch.
# This is not the usual 'visit every example once per epoch' training order.
class InfiniteDataLoader:


    # **kwargs collects additional named arguments into a dictionary, for
    # example {batch_size:32,num_workers:4} with string keys. They will be
    # forwarded to DataLoader rather than handled separately by this wrapper.
    def __init__(self, dataset, **kwargs):
        # Request ten billion random example selections. 1e10 is scientific
        # notation; int converts it to an integer count. replacement=True allows
        # the same example to be selected repeatedly. Indices are produced by the
        # sampler over time, rather than this line constructing all examples at once.
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        # Build a batch loader using that sampler. **kwargs unpacks a dictionary
        # of extra named settings, such as batch_size and num_workers, into this call.
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        # Create an iterator: an object that remembers progress and supplies
        # one next batch at a time. Iterators are advanced with next(...).
        self.data_iter = iter(self.train_loader)

    # Return another batch, restarting the underlying iterator if it is exhausted.
    def next(self):
        # Attempt the indented operation. If it raises a matching exception,
        # execution jumps to the corresponding except block.
        try:
            # Ask the current iterator for the next input/target batch.
            batch = next(self.data_iter)
        # Catch the specific signal that an iterator has no more batches.
        # Other errors are not swallowed by this handler.
        except StopIteration:
            # Create an iterator: an object that remembers progress and supplies
            # one next batch at a time. Iterators are advanced with next(...).
            self.data_iter = iter(self.train_loader)
            # Ask the current iterator for the next input/target batch.
            batch = next(self.data_iter)
        # Return the batch to the training loop.
        return batch


# PROGRAM ENTRY POINT. Python assigns __name__="__main__" when you run
# this file directly. Importing it defines the classes/functions but skips
# this training block. The guard also matters for spawned data workers.
# For example, from this repository folder a command can select a small
# run: python makemore.py --type bigram --max-steps 100 --num-workers 0
# The relative default names.txt and out paths refer to your current folder.
if __name__ == '__main__':


    # Create the command-line option parser. Running with --help displays
    # the option names and their existing help strings.
    parser = argparse.ArgumentParser(description="Make More")

    # Path to a text file with one word per line; -i is a shorter spelling.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    # Output directory for checkpoints and event logs; -o is an alias.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    # A presence-only flag: store_true makes its value True when included.
    # It reloads model weights, then starts a fresh optimizer and counter.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    # A Boolean presence flag: load the checkpoint, print names, then exit.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    # Convert the supplied value to int. Controls training loader workers.
    # Zero means load batches in the main process.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    # Integer limit for training updates; default -1 keeps looping.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    # String selecting compute hardware. cpu is the default; other devices
    # require corresponding hardware and PyTorch support.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    # Integer seed controlling pseudorandom initialization, shuffling, sampling.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--seed', type=int, default=3407, help="seed")

    # Integer sampling cutoff; -1 disables filtering. Otherwise use 1..V.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")

    # Select one of the six model architectures by its string name.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    # Transformer depth: how many independent attention blocks are stacked.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    # Attention head count; it must divide n_embd evenly.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    # Embedding feature width C; a larger width increases model capacity and cost.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    # Second hidden width S, used by recurrent, MLP, and BoW components.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")

    # Number of words sampled for one optimizer update, not character count.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    # Convert input to float. Default 5e-4 means 0.0005, controlling update size.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    # AdamW shrinkage strength. Default 0.01 encourages smaller parameters.
    # In this call, type converts input, default supplies a missing value,
    # and help supplies the explanation shown by --help (where applicable).
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    # Read command-line options into an object. Hyphens become underscores:
    # --batch-size 32 is available as args.batch_size. Missing options use defaults.
    args = parser.parse_args()
    # vars converts the settings object to a dictionary so you can see the run settings.
    print(vars(args))


    # Seed PyTorch random generators to make repeated runs more comparable.
    # A seed selects a reproducible pseudorandom sequence; it does not remove
    # randomness or guarantee identical results across all hardware/software.
    torch.manual_seed(args.seed)
    # Seed random generators for CUDA GPUs as well. CUDA is PyTorch support
    # for compatible NVIDIA accelerators; CPU is the default compute device here.
    torch.cuda.manual_seed_all(args.seed)
    # Create the output folder and missing parent folders. exist_ok=True
    # allows an already-existing directory instead of raising that error.
    os.makedirs(args.work_dir, exist_ok=True)
    # Create TensorBoard event logging in the output folder for later loss charts.
    writer = SummaryWriter(log_dir=args.work_dir)


    # Read, clean, split, and wrap the words. Tuple unpacking stores both results.
    train_dataset, test_dataset = create_datasets(args.input_file)
    # Get V, including the shared START/STOP token, from the dataset.
    vocab_size = train_dataset.get_vocab_size()
    # Use longest-word length plus one as the maximum context length.
    block_size = train_dataset.get_output_length()
    # Display these values for the person running the program. f-strings
    # insert expressions inside braces; this line reports progress or dataset details.
    print(f"dataset determined that: {vocab_size=}, {block_size=}")


    # Construct the common settings object. The next two continued lines
    # copy architecture choices from args; parentheses permit multiline calls.
    # Not every model uses every field (e.g. bigram ignores embedding widths).
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    if args.type == 'transformer':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = Transformer(config)
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    elif args.type == 'bigram':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = Bigram(config)
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    elif args.type == 'mlp':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = MLP(config)
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    elif args.type == 'rnn':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = RNN(config, cell_type='rnn')
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    elif args.type == 'gru':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = RNN(config, cell_type='gru')
    # Select this architecture only when its name matches args.type.
    # The if/elif chain chooses one branch, so only one model is constructed.
    elif args.type == 'bow':
        # Create the selected model with fresh trainable parameters. The shared
        # interface lets the same training and sampling code work with this choice.
        model = BoW(config)
    # Fallback branch: run this only when the preceding if/elif conditions fail.
    else:
        # Stop with an explicit error for an unrecognized architecture name.
        # raise signals an exception rather than returning a normal result.
        raise ValueError(f'model type {args.type} is not recognized')
    # Move registered parameters and buffers to the requested device.
    # Input and target batches must also be moved there before calculations.
    model.to(args.device)
    # Count ALL parameters in the selected model, including any output head.
    # This is the number of trainable scalar values, not the number of layers.
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    # or is true when either flag is enabled. Sampling-only also loads saved
    # weights; a newly initialized random model would otherwise generate nonsense.
    if args.resume or args.sample_only:
        # Display these values for the person running the program. f-strings
        # insert expressions inside braces; this line reports progress or dataset details.
        print("resuming from existing model in the workdir")
        # Join the folder and checkpoint filename, load saved tensors, then copy
        # them into the model. The chosen architecture and vocabulary mapping must
        # match the saved run. Only model state is saved here: resume does NOT
        # restore optimizer momentum, prior step count, or the previous best loss.
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    # If requested, generate samples from loaded weights and exit before training.
    if args.sample_only:
        # Display 50 randomly generated sequences, grouped by exact dataset matches.
        print_samples(num=50)
        # End execution here so the optimizer and training loop are skipped.
        sys.exit()


    # Create AdamW, an optimizer that updates all registered model parameters.
    # For gradient g, it tracks smoothed first/second moments:
    # m <- 0.9*m + 0.1*g; v <- 0.99*v + 0.01*g*g. Bias-corrected versions
    # m_hat and v_hat compensate for their initial zeros. The adaptive update
    # is approximately parameter -= lr*m_hat/(sqrt(v_hat)+eps), plus separate
    # weight decay parameter *= (1-lr*weight_decay). AdamW applies that decay
    # to this entire parameter list, including biases and normalization weights.
    # lr controls update size; eps=1e-8 stabilizes division. The optimizer
    # object stores this history, but does not update anything until step().
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)


    # Create the repeating training-batch source. batch_size counts names
    # per update; num_workers controls loading subprocesses. pin_memory asks
    # for host memory suitable for faster accelerator transfers where supported.
    # Only training uses this worker count; evaluate uses zero workers.
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)


    # No held-out checkpoint score has been recorded in this run yet.
    best_loss = None
    # Start the update counter at zero, including when resuming weights.
    step = 0
    # Repeat indefinitely until the explicit break condition at the bottom.
    # With default max_steps=-1, you stop the process externally when finished.
    while True:

        # Record the start time in seconds to estimate training iteration duration.
        t0 = time.time()


        # Fetch another randomly sampled batch of name-prediction exercises.
        batch = batch_loader.next()
        # List comprehension: move each tensor t in the batch to the chosen
        # compute device. It returns a new list holding the moved input and target.
        batch = [t.to(args.device) for t in batch]
        # Unpack the batch: X is input IDs and Y is the desired next-character IDs.
        # Both have shape (batch_size, sequence_length), aside from a shorter final batch.
        X, Y = batch


        # Calling an nn.Module invokes its forward method through PyTorch.
        # Supply input AND targets to obtain predictions and cross-entropy loss.
        # During training a gradient graph is built; evaluation decorators disable it.
        logits, loss = model(X, Y)


        # Clear gradients from the previous update. PyTorch normally accumulates
        # into .grad, so skipping this would mix steps unintentionally.
        # set_to_none=True removes stored gradients instead of filling them with zeros;
        # it does not erase the learned parameter values.
        model.zero_grad(set_to_none=True)
        # Backpropagation: walk the recorded forward computation backward.
        # The chain rule multiplies local derivatives along paths and sums effects
        # where a parameter influences loss more than once. Each trainable tensor
        # receives d(loss)/d(parameter) in .grad. No weights change on this line.
        loss.backward()
        # Use the current gradients and AdamW history to change trainable values.
        # A positive gradient suggests decreasing that parameter locally; a negative
        # one suggests increasing it. This aims to reduce loss but does not guarantee
        # that every individual update improves held-out predictions.
        optimizer.step()


        # GPU operations can run asynchronously. Only synchronize for CUDA here.
        if args.device.startswith('cuda'):
            # Wait for CUDA work to finish so the end time includes GPU execution.
            torch.cuda.synchronize()
        # Record the end time. t1-t0 measures this update, excluding evaluation
        # and sample printing performed farther down the loop.
        t1 = time.time()


        # Remainder zero means step is divisible by 10; log at steps 0,10,20,...
        if step % 10 == 0:
            # Print the PRE-update forward-pass loss and elapsed step time. .4f
            # formats four decimals, .2f two; multiplying seconds by 1000 gives ms.
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")


        # Every 500 counter steps after zero, estimate train and test error.
        # Because the counter starts at zero and updates happen before this check,
        # step 500 follows the 501st optimizer update.
        if step > 0 and step % 500 == 0:
            # Measure prediction error on a shuffled subset of training examples,
            # without changing weights. This call can process up to 11 batches.
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            # Measure error on held-out examples. Lower training loss with rising
            # test loss can indicate overfitting: memorizing training patterns that
            # do not generalize well. This split is also used to choose checkpoints,
            # so it acts as a validation set rather than an untouched final benchmark.
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            # Record the training estimate under its chart label and current counter.
            writer.add_scalar("Loss/train", train_loss, step)
            # Record the held-out estimate for comparison with training loss.
            writer.add_scalar("Loss/test", test_loss, step)
            # Push pending TensorBoard log records to disk.
            writer.flush()
            # Display these values for the person running the program. f-strings
            # insert expressions inside braces; this line reports progress or dataset details.
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")

            # Save on the first evaluation or when held-out loss improves. or
            # short-circuits: if best_loss is None, Python skips the numeric comparison.
            if best_loss is None or test_loss < best_loss:
                # Build the checkpoint path using the operating system path separator.
                out_path = os.path.join(args.work_dir, "model.pt")
                # Display these values for the person running the program. f-strings
                # insert expressions inside braces; this line reports progress or dataset details.
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                # Save named model parameter tensors and registered buffers, overwriting
                # this checkpoint path. This does not save the dataset character mapping,
                # configuration, optimizer state, or current step. The first save occurs
                # at evaluation time; a short run may finish without creating model.pt.
                torch.save(model.state_dict(), out_path)
                # Remember this score as the comparison point for later checkpoint decisions.
                best_loss = test_loss


        # Every 200 counter steps after zero, print names to inspect learned patterns.
        if step > 0 and step % 200 == 0:
            # Generate ten random examples with the current weights; no training update
            # occurs inside this helper.
            print_samples(num=10)

        # Increase the counter by one. += is shorthand for step = step + 1.
        step += 1

        # Stop when a nonnegative requested limit is reached. -1 disables this
        # limit. The check is after an update, so even max_steps=0 runs one update
        # in this existing implementation. Stopping does not itself save a checkpoint.
        if args.max_steps >= 0 and step >= args.max_steps:
            # Exit the nearest enclosing loop immediately, continuing after that loop.
            break
