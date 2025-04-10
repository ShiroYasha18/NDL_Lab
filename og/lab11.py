import torch, torch.nn.functional as F
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset

X, y = load_digits(return_X_y=True)
X = torch.tensor(X.reshape(-1, 1, 8, 8) / 16).float(); y = torch.tensor(y)
data = DataLoader(TensorDataset(X, y), 64)
model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3), torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(16*6*6, 10))
optim = torch.optim.Adam(model.parameters())

for xb, yb in data: optim.zero_grad(); F.cross_entropy(model(xb), yb).backward(); optim.step()
print("Acc:", (model(X).argmax(1) == y).float().mean().item())
