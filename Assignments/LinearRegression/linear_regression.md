---
jupytext:
  cell_metadata_json: true
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text
```

+++ {"slideshow": {"slide_type": "slide"}}

## Linear regression

+++

In this notebook we will consider a simple linear regression model:

+++ {"slideshow": {"slide_type": "fragment"}}

$$ y_i = x_{ij} w_j + b$$

+++

We will be using the "summation conventions": when an index is repeated the summation over this index is implied:

+++

$$ 
x_{ij} w_j \equiv   \sum_j x_{ij} w_j 
$$

+++

#### Problem 1

+++

Implement function `linear(x,w,b)` that given feature matrix $\mathbf{x}$, weights $\mathbf{w}$ and bias $b$  returns $\mathbf{y}$. **Hint** Use matrix multiplication operator `@`.

```{code-cell} ipython3
def linear(x,w,b):
    return x @ w + b
```

### Data

+++

#### Problem 2

+++ {"slideshow": {"slide_type": "-"}}

Generate a random feature matrix $\mathbf{x}$ witch 10000 samples and three features, such that first feature is drawn from normal distribution $\mathcal{N}(0,1)$, second feature from  uniform distribution on interval $[0,1)$ and third from $\mathcal{N}(1,2)$, where 
$N(\mu,\sigma)$ denotes normal distribution with mean $\mu$ and standard deviation $\sigma$. To generate random numbers you can use `numpy.random.normal` and `numpy.random.uniform` functions. To collect all features together you can use `numpy.stack` function.

+++ {"slideshow": {"slide_type": "-"}}

Then using $\mathbf{x}$, weights $w_{true}$  and  bias $b_{true}$:

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
w_true = np.array([0.2, 0.5,-0.2])
b_true = -1

N_normal = np.random.normal(0, 1, 10000)
N_01 = np.random.uniform(0, 1, 10000)
N_12 = np.random.normal(1, 2, 10000)
X = np.stack((N_normal, N_01, N_12), axis=-1)
print(X.shape)
```

+++ {"slideshow": {"slide_type": "-"}}

generate output $\mathbf{y}$ assuming a normaly distributed $\mathcal{N}(0,0.1)$ noise $\mathbf{\epsilon}$.

+++ {"slideshow": {"slide_type": "-"}}

$$ y_i =  
x_{ij} w_j+b +\epsilon_i 
$$

```{code-cell} ipython3
y = linear(X, w_true, b_true) + np.random.normal(0, .1, 10000)
y.shape
```

+++ {"slideshow": {"slide_type": "slide"}}

### Loss

+++

#### Problem 3

+++

Given the means square loss

+++ {"slideshow": {"slide_type": "fragment"}}

$$ MSE(w,b|y,x) = \frac{1}{2}\frac{1}{N}\sum_{i=0}^{N-1} (y_i -  x_{ij} w_j -b  )^2$$

+++

write down the python function `mse(y,x,w,b)` implementing it:

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
tags: []
---
def mse(y,x,w,b):
    return ((y - linear(x, w, b)) ** 2).mean()
```

```{code-cell} ipython3
mse(y,X,w_true,b_true)
```

### Gradient

+++

and implement functions `grad_w(y,x,w,b)` and `grad_b(y,x,w,b)` implementing those gradients.

```{code-cell} ipython3
:tags: []

def grad_w(y,x,w,b):
    n = x.shape[0]
    return -2/n * (y @ x - x.T @ x @ w - b * x.sum(0))

def grad_b(y,x,w,b):
    n = x.shape[0]
    return -2/n * (y - x @ w - b).sum()
```

```{code-cell} ipython3
grad_w(y,X,w_true,b_true)
```

```{code-cell} ipython3
grad_b(y,X,w_true,b_true)
```

### Gradient descent

+++ {"tags": []}

#### Problem 4

+++

Implement gradient descent for linear regression. Start from

```{code-cell} ipython3
w = np.asarray([0.0,0.0,0.0], dtype='float64')
b = 1.0 
lr = 0.1 #learning rate
epochs = 1000
err = 0.0075

for i in range(epochs):
    error = mse(y,X,w,b)
    if (error < err):
        print(i)
        break
    w -= (lr * grad_w(y,X,w,b))
    b -= (lr * grad_b(y,X,w,b))
    
print(w, b)
print(w_true, b_true)
```

How many epochs did you need to get MSE below 0.0075 ?

```{code-cell} ipython3
print(error)
```

### Pytorch

+++

#### Problem 5

+++

Implement gradient descent using pytorch. Start by just rewritting Problem 4 to use torch Tensors instead of numpy arrays.

+++

To convert frrom numpy arrays to torch tensors you can use ``torch.from_numpy()`` function:

```{code-cell} ipython3
import torch as t 
```

```{code-cell} ipython3
t_y = t.from_numpy(y)
t_x = t.from_numpy(X)
t_w = t.DoubleTensor([0,0,0])
t_b = t.DoubleTensor([1.0])
```

Then use the automatic differentiation capabilities of Pytorch. To this end the variable with respect to which the gradient will be calculated, `t_w` and `t_b` in this case, must have attribute
`requires_grad` set to `True`.

```{code-cell} ipython3
t_w.requires_grad_(True)
t_b.requires_grad_(True)
print(t_w, t_b)
```

```{code-cell} ipython3
epochs = 1000
eta = 0.1
for i in range(epochs):
    if not (t_w.grad is None):
        t_w.grad.data.zero_()
    if not (t_b.grad is None):
        t_b.grad.data.zero_()
    loss = mse(t_y,t_x,t_w,t_b)
    loss.backward()
    t_w.data.sub_(eta*t_w.grad)
    t_b.data.sub_(eta*t_b.grad)
    
print(t_w, t_b)
print(w_true, b_true)
```

```{code-cell} ipython3
print(loss)
```

The torch will automatically track any expression containing `t_w` and `t_b` and store its computational graph. The method `backward()` can be run on the final expression to back propagate the gradient. The gradient is then accesible as `t_w.grad`.

+++

Finally use  Pytorch  optimisers.
