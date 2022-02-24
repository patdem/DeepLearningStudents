---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import scipy
import scipy.stats
import torch as t

import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout, Sigmoid
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
device=t.device('cpu') #Overrride the above device choice
```

Generate the sample 2D distribution: uniform from unit circle.

```{code-cell} ipython3
angle = np.random.uniform(-np.pi,np.pi,(1024,1)).astype('float32')
data = np.concatenate((np.cos(angle), np.sin(angle)),axis=1)
```

```{code-cell} ipython3
plt.scatter(data[:,0], data[:,1]);
```

```{code-cell} ipython3
data_t = t.from_numpy(data)
```

```{code-cell} ipython3
data_t
```

```{code-cell} ipython3
batch_size = 32
train_loader = t.utils.data.DataLoader(
    data_t, batch_size=batch_size, shuffle=True
)
```

```{code-cell} ipython3
class Discriminator(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = t.nn.Sequential(
            t.nn.Linear(2, 256),
            t.nn.ReLU(),
            t.nn.Dropout(0.3),
            t.nn.Linear(256, 128),
            t.nn.ReLU(),
            t.nn.Dropout(0.3),
            t.nn.Linear(128, 64),
            t.nn.ReLU(),
            t.nn.Dropout(0.3),
            t.nn.Linear(64, 1),
            t.nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output
```

```{code-cell} ipython3
discriminator = Discriminator()
```

```{code-cell} ipython3
discriminator = discriminator.to(device) 
```

```{code-cell} ipython3
class Generator(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = t.nn.Sequential(
            t.nn.Linear(2, 16),
            t.nn.ReLU(),
            t.nn.Linear(16, 32),
            t.nn.ReLU(),
            t.nn.Linear(32, 2),
        )
    def forward(self, x):
        output = self.model(x)
        return output
```

```{code-cell} ipython3
generator = Generator()
```

```{code-cell} ipython3
generator = generator.to(device)
```

```{code-cell} ipython3
lr = 0.001
num_epochs = 300
loss_function = t.nn.BCELoss()
```

```{code-cell} ipython3
optimizer_discriminator = t.optim.Adam(discriminator.parameters(), lr=lr)
```

```{code-cell} ipython3
optimizer_generator = t.optim.Adam(generator.parameters(), lr=lr)
```

### Problem 1

+++

Implement the GAN train loop that will train GAN to generate from the sample distribution.

+++

Update to Pegaz both the notebook and the trained generator.

```{code-cell} ipython3
for epoch in range(num_epochs):
    for n, real_samples in enumerate(train_loader):
        # Data for training the discriminator
        # labels with value of 1 for real lables
        real_samples_labels = t.ones((batch_size, 1))
        # fake data - some randoms
        fake_samples = t.randn((batch_size, 2))
        # feed generator with fake data
        generated_samples = generator(fake_samples)
        # labels with value of 0 for fake labels
        generated_samples_labels = t.zeros((batch_size, 1))
        # concatenated samples - real and fake - for the discriminator
        all_samples = t.cat((real_samples, generated_samples))
        all_samples_labels = t.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        # clear the gradients
        discriminator.zero_grad()
        # feed the dicriminator with all samples
        output_discriminator = discriminator(all_samples)
        # calculate the loss
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        # calculate the gradients to update the weights
        loss_discriminator.backward()
        # update the weights
        optimizer_discriminator.step()

        # Data for training the generator
        fake_samples = t.randn((batch_size, 2))

        # Training the generator
        # clear the gradients
        generator.zero_grad()
        # feed the generator with fake data
        generated_samples = generator(fake_samples)
        # feed the discriminator with generator's samples 
        output_discriminator_generated = discriminator(generated_samples)
        # calculate the loss
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        # calculate the gradients to update the weights
        loss_generator.backward()
        # update the weights
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
```

```{code-cell} ipython3
# test the generator
fake_samples = t.randn(1000, 2)
generated_samples = generator(fake_samples)
```

```{code-cell} ipython3
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
```

```{code-cell} ipython3
t.save(generator.state_dict(), "./generator.pt")
```

### Problem 2

+++

Use sampling distribution below.

```{code-cell} ipython3
n_samples = 10240
a = 2
b = 1
angle2 = np.random.uniform(-np.pi,np.pi,(n_samples,1)).astype('float32')
r = np.sqrt(np.random.uniform(0.5,1,(n_samples,1)))
data2 = np.stack((a*r*np.cos(3*angle2), b*r*np.sin(2*angle2)),axis=1)
```

```{code-cell} ipython3
plt.scatter(data2[:,0], data2[:,1], s=2, alpha=0.5);
```

```{code-cell} ipython3
data_t2 = t.from_numpy(data2)
```

```{code-cell} ipython3
data_t2.shape
```

```{code-cell} ipython3
data_t2
```

```{code-cell} ipython3
train_loader2 = t.utils.data.DataLoader(
    data_t2, batch_size=batch_size, shuffle=True
)
```

```{code-cell} ipython3
class Discriminator2(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = t.nn.Sequential(
            t.nn.Linear(1, 128),
            t.nn.ReLU(),
            t.nn.Dropout(0.3),
            t.nn.Linear(128, 64),
            t.nn.ReLU(),
            t.nn.Dropout(0.3),
            t.nn.Linear(64, 1),
            t.nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(x.size(0), 2, 1)
        output = self.model(x)
        return output
```

```{code-cell} ipython3
discriminator2 = Discriminator2()
```

```{code-cell} ipython3
class Generator2(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = t.nn.Sequential(
            t.nn.Linear(1, 64),
            t.nn.ReLU(),
            t.nn.Linear(64, 32),
            t.nn.ReLU(),
            t.nn.Linear(32, 1),
        )
    def forward(self, x):
        x = x.view(x.size(0), 2, 1)
        output = self.model(x)
        return output
```

```{code-cell} ipython3
generator2 = Generator2()
```

```{code-cell} ipython3
lr = 0.001
num_epochs = 300
loss_function = t.nn.BCELoss()
```

```{code-cell} ipython3
optimizer_discriminator2 = t.optim.Adam(discriminator2.parameters(), lr=lr)
```

```{code-cell} ipython3
optimizer_generator2 = t.optim.Adam(generator2.parameters(), lr=lr)
```

```{code-cell} ipython3
for epoch in range(num_epochs):
    for n, real_samples in enumerate(train_loader2):
        # Data for training the discriminator
        # labels with value of 1 for real lables
        real_samples_labels = t.ones((batch_size, 1))
        # fake data - some randoms
        fake_samples = t.randn((batch_size, 2, 1))
        # feed generator with fake data
        generated_samples = generator2(fake_samples)
        # labels with value of 0 for fake labels
        generated_samples_labels = t.zeros((batch_size, 1))
        # concatenated samples - real and fake - for the discriminator
        all_samples = t.cat((real_samples, generated_samples))
        all_samples_labels = t.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        # clear the gradients
        discriminator2.zero_grad()
        # feed the dicriminator with all samples
        output_discriminator = discriminator2(all_samples)
        # calculate the loss
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        # calculate the gradients to update the weights
        loss_discriminator.backward()
        # update the weights
        optimizer_discriminator2.step()

        # Data for training the generator
        fake_samples = t.randn((batch_size, 2, 1))

        # Training the generator
        # clear the gradients
        generator2.zero_grad()
        # feed the generator with fake data
        generated_samples = generator2(fake_samples)
        # feed the discriminator with generator's samples 
        output_discriminator_generated = discriminator2(generated_samples)
        # calculate the loss
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        # calculate the gradients to update the weights
        loss_generator.backward()
        # update the weights
        optimizer_generato2r.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
```

Here I got an error: RuntimeError: expected scalar type Double but found Float in line 22. I can't solve this problem.

+++

Update to Pegaz both the notebook and the trained generator.
