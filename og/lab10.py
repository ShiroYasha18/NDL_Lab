import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X, y = torch.tensor(X/16).float(), torch.tensor(y).long()  # Scale data 0-1

for opt_name in ['Adagrad', 'RMSprop', 'Adam']:
    model = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,10))
    opt = getattr(optim, opt_name)(model.parameters())
    for _ in range(5):  # 5 epochs
        for xb, yb in zip(X.split(64), y.split(64)):  # Manual batches
            opt.zero_grad(); nn.functional.cross_entropy(model(xb), yb).backward();opt.step()

    print(f"{opt_name}: {(model(X).argmax(1) == y).float().mean():.2f}")