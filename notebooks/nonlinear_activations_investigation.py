# %%
# ruff: noqa: N803, N806 # allow cap vars to represent matrices
import math
from typing import Callable

import numpy as np
from scipy.special import erf

import matplotlib.pyplot as plt


# %%
def sigmoid(x: np.ndarray, beta: float = 1.0):
    """Parameterized sigmoid with learnable beta."""
    return 1 / (1 + np.exp(-beta * x))


def tanh(x: np.ndarray):
    return np.tanh(x)


def relu(x: np.ndarray):
    """Rectified Linear Unit."""
    return np.maximum(x, 0)


def leakyrelu(x: np.ndarray):
    """Leaky ReLU."""
    return np.maximum(x, 0.01 * x)


def gelu(x: np.ndarray):
    """Gaussian-error linear unit."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def swish(x: np.ndarray, beta: float = 1.0):
    """SiLU (Sigmoid Linear Unit) aka Swish."""
    return x * sigmoid(x, beta)


# NOTE:
# GLU (Gated Linear Units) are technically a type of neural network layer, not an activation function in the strict sense.
# It is a linear transformation followed by a gating mechanism.
#
# def swiglu(x: np.ndarray, beta: float = 1):
#     """Swish-Gated Linear Unit."""
#     return np.dot(swish(xW, b), xV+c)


# %%
def dfdx(f: Callable, x: np.ndarray):
    return np.gradient(f(x), x)


def plot_derivative(ax, f: Callable, x: np.ndarray):
    ax = ax or plt.gca()

    ax.axvline(x=0, color="black", linewidth=0.25)
    ax.axhline(y=0, color="black", linewidth=0.25)
    ax.plot(x, f(x), label="f(x)")
    ax.plot(x, dfdx(f, x), label="df/dx")

    ax.set_title(f.__name__)
    ax.legend()
    # plt.show()
    return ax


# %% [markdown]
# ## Plot Activations & Derivatives

# %%
x = np.linspace(-4, 4, 1000)
functions = [sigmoid, tanh, relu, leakyrelu, gelu, swish]
fig, axes = plt.subplots(2, math.ceil(len(functions) / 2), sharex=True, sharey=True, figsize=(16, 8))

for ax, f in zip(axes.ravel(), functions):
    plot_derivative(ax, f, x)

plt.show()

# %% [markdown]
# ### Commentary - Nonlinearities and derivatives
#
# From [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) and [CS231n](https://www.youtube.com/watch?v=gYpoJMlgyXA&t=1170s)
#
# #### sigmoid
#
# 1. saturated neurons "kill" the gradients
# 2. sigmoid outputs are not 0-centered
# 3. `exp` is kind of expensive
#
# #### tanh
#
# 1. _is_ 0-centered
# 2. still kills gradients
#
# #### ReLU
#
# 1. does _not_ saturate (in + region)
# 2. computationally efficient
# 3. converges ~6x faster than sigmoid/tanh
# 4. not 0-centered
#
# > diagnostic: if relu killed neurons during training, learning rate is too high (or weights were poorly initialized)
#
# #### Leaky ReLU
#
# 1. All the benefits of ReLU, but leaky negative region doesn't kill neurons
#
# #### GELU
#
# 1. Used in BERT, GPT-2, GPT-3
# 2. Can consider as smooth version of ReLU
#
# #### Swish / SiLU
#
# 1. Consider as smooth version of ReLU
# 2. Very similar to GELU, but has learnable param beta


# %% [markdown]
# ## What do these derivatives tells us about deep networks?
#
# Can we figure out why vanishing/exploding gradients are a problem?
# Can we figure out how to resolve?


# %%
def assess_activations(
    batchdims: tuple[int] = (1000, 500),
    ndim: int = 500,
    nlayers: int = 10,
    init_scale: float = 0.01,
    fn: Callable = tanh,
    seed: int = 1337,
):
    """Assess implications of deep networks with repeated activation fn.
    Forward pass only.
    """
    # set seed to run this experiment repeatably
    rng = np.random.default_rng(seed)

    # init input data
    D = rng.random(batchdims)
    hidden_layer_sizes = [ndim] * nlayers

    Hs = {}
    for i, _ in enumerate(hidden_layer_sizes):
        X = D if i == 0 else Hs[i - 1]  # input at this layer
        fan_in = X.shape[1]
        fan_out = hidden_layer_sizes[i]
        W = rng.standard_normal((fan_in, fan_out)) * init_scale  # layer initialization

        H = np.dot(X, W)  # matrix multiply
        H = fn(H)  # nonlinearity
        Hs[i] = H  # cache result on this layer

    # look at distributions at each layer
    print(f"input layer had mean {np.mean(D)} and std {np.std(D)}")
    layer_means = [np.mean(H) for H in Hs.values()]
    layer_stds = [np.std(H) for H in Hs.values()]
    for i, _ in enumerate(Hs):
        print(f"Hidden layer {i + 1} had mean of {layer_means[i]} and std {layer_stds[i]}")

    # plot the means and standard deviations
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(Hs.keys(), layer_means, "ob-")
    ax[0].set_title("layer mean")
    ax[1].plot(Hs.keys(), layer_stds, "or-")
    ax[1].set_title("layer std")
    plt.show()

    # plot the raw distributions
    fig, ax = plt.subplots(1, len(Hs), sharey=True, figsize=(16, 8))
    for i, H in Hs.items():
        # plt.subplot(1, len(Hs), i + 1, sharey=True)
        ax[i].hist(H.ravel(), 30, range=(-1, 1))


# %% [markdown]
# ### Commentary - Implications of Deep Networks

# %%
# assume some unit gaussian 10-0 input data
batchdims = (1000, 500)
ndim = 500  # neurons / layer
nlayers = 10

x = np.linspace(-4, 4, 1000)

# %% [markdown]
# #### tanh

# %%
fig, ax = plt.subplots(figsize=(8, 6))
plot_derivative(ax, tanh, x)
plt.show()

# %% [markdown]
# ##### tanh with small weight inits (0.01)
#
# 1. Deep layers rapidly tend to 0
# 2. Means gradients are almost flat (no Δ) - very slow training
# 3. Means backprop changes all weights to ~0

# %%
assess_activations(
    batchdims=batchdims,
    ndim=ndim,
    nlayers=nlayers,
    fn=tanh,
    init_scale=0.01,
)

# %% [markdown]
# ##### tanh with large weight inits (1.0)
#
# 1. All layers are completely saturated (all -1 or all 1) -- bimodal
# 2. Means gradients are 0 the entire time

# %%
assess_activations(
    batchdims=batchdims,
    ndim=ndim,
    nlayers=nlayers,
    fn=tanh,
    init_scale=1.0,
)


# %% [markdown]
# ##### Xavier initalization
#
# 1. scales weights based on neurons in network

# %%
assess_activations(
    batchdims=batchdims,
    ndim=ndim,
    nlayers=nlayers,
    fn=tanh,
    init_scale=1 / np.sqrt(ndim),
)  # Xavier initialization


# %% [markdown]
# #### ReLU

# %%
fig, ax = plt.subplots(figsize=(8, 6))
plot_derivative(ax, relu, x)
plt.show()

# %% [markdown]
# ##### Xavier initalization
#
# 1. Deep layers rapidly tend to 0 (half the positive neurons each time)

# %%
assess_activations(
    batchdims=batchdims,
    ndim=ndim,
    nlayers=nlayers,
    fn=relu,
    init_scale=1 / np.sqrt(ndim),
)  # Xavier initialization


# %% [markdown]
# ##### Xavier initalization (fixed for ReLU)
# 2. Scale Xavier initialization by 2 to account to ^

# %%
assess_activations(
    batchdims=batchdims,
    ndim=ndim,
    nlayers=nlayers,
    fn=relu,
    init_scale=1 / np.sqrt(ndim / 2),
)  # Xavier initialization fixed for relu


# %% [markdown]
# > These issues are mostly resolved with batch-norm
# > As seen above, we _want_ our weights to be roughly unit gaussian.
# >
# > Batch norm adds a layer that takes the weights and normalizes them (for each batch).

# %%
