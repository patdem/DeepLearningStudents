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

# Skin segmentation

+++

In this assignement you will train classifier to assign colors to skin or no skin classes. The data is taken from [Skin Segmentation Data Set](http://archive.ics.uci.edu/ml/datasets/Skin+Segmentation#) in the UCI Machine Learning repository.

+++

The  data is in a plain text format and contains four columns. First three contain RGB color data  represented as integers in the range 0-255, and the last column is an integer label  with 1 representing skin and 2 representing no skin. This file we can load directly into a numpy array:

```{code-cell} ipython3
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
```

```{code-cell} ipython3
data = np.loadtxt('data/Skin_NonSkin.txt')
```

```{code-cell} ipython3
rgb = data[:,:3].astype('float32')
lbl = data[:,3].astype('float32') 
lbl = 2-lbl
```

```{code-cell} ipython3
len(rgb)
```

```{code-cell} ipython3
len(lbl)
```

```{code-cell} ipython3
np.bincount(lbl.astype('int32'))
```

## Problem 1

+++

Train the neural network to distinguish skin from no skin colors. Calculate the accuracy on train and validation sets. Calculate true positives rate and false positives rate.

```{code-cell} ipython3
rgb
```

```{code-cell} ipython3
lbl
```

```{code-cell} ipython3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(rgb, lbl, test_size=0.3, random_state=RANDOM_SEED)
```

```{code-cell} ipython3
X_train = torch.from_numpy(X_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())
X_test = torch.from_numpy(X_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test).float())

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
```

```{code-cell} ipython3
class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 5)
    self.fc2 = nn.Linear(5, 3)
    self.fc3 = nn.Linear(3, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))

net = Net(X_train.shape[1])
```

```{code-cell} ipython3
criterion = nn.BCELoss()
```

```{code-cell} ipython3
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

```{code-cell} ipython3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)
```

```{code-cell} ipython3
def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)
```

```{code-cell} ipython3
def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(1000):
    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
      train_acc = calculate_accuracy(y_train, y_pred)
      y_test_pred = net(X_test)
      y_test_pred = torch.squeeze(y_test_pred)
      test_loss = criterion(y_test_pred, y_test)
      test_acc = calculate_accuracy(y_test, y_test_pred)
      print(
f'''epoch {epoch}
Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
```

```{code-cell} ipython3
y_pred = net(X_test)
y_pred = y_pred.ge(.5).view(-1).cpu()
y_test = y_test.cpu()
print(classification_report(y_test, y_pred))
```

```{code-cell} ipython3
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

```{code-cell} ipython3
tpr = tp / (tp+fn)
print(f"True positve rate = {tpr}")
```

```{code-cell} ipython3
fpr = fp / (fp+tn)
print(f"False positve rate = {fpr}")
```
