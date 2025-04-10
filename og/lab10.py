import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

X, y = load_digits(return_X_y=True)
X, y = torch.tensor(X / 16).float(), torch.tensor(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
train_loader = DataLoader(TensorDataset(X_train, y_train), 64)

class Net(nn.Module):
    def __init__(self): super().__init__(); self.l1 = nn.Linear(64, 32); self.l2 = nn.Linear(32, 10)
    def forward(self, x): return self.l2(F.relu(self.l1(x)))

for opt_name in ['Adagrad', 'RMSprop', 'Adam']:
    model = Net(); optim = getattr(torch.optim, opt_name)(model.parameters())
    for _ in range(5):
        for xb, yb in train_loader:
            optim.zero_grad(); F.cross_entropy(model(xb), yb).backward(); optim.step()
    acc = (model(X_test).argmax(1) == y_test).float().mean()
    print(f"{opt_name} accuracy: {acc.item():.4f}")
