import os
os.path.join("C:/Users/user/Python_Projects/ViT")

from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets.mnist import MNIST
from vision_transformer import MyViT

# plt.imshow(positional_embedding(100, 300), cmap="hot", interpolation="nearest")
# plt.show()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn():
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Lambda(lambda x: x.type(torch.uint8)),
      torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31)

  ])

  train_set = MNIST(root="./../datasets", train=True, download=True, transform=transform)
  test_set = MNIST(root="./../datasets", train=False, download=True, transform=transform)

  train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
  test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False)

  model = MyViT((1, 28, 28)).to(DEVICE)

  n_epochs = 1
  lr = 0.005

  loss_fn = CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=lr)

  for epoch in trange(n_epochs, desc = "Training"):
    train_loss = 0
    train_acc = 0

    test_loss = 0
    test_acc = 0

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
      X, y = X.to(DEVICE), y.to(DEVICE)

      y_logits = model(X).to(DEVICE)
      loss = loss_fn(y_logits, y)

      y_pred = torch.argmax(y_logits, dim = 1)

      train_loss += loss.detach().cpu().item()
      train_acc += (y_pred == y).sum().item() / len(y)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    model.eval()
    with torch.no_grad():
      for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        y_logits = model(X).to(DEVICE)
        y_pred = torch.argmax(y_logits, dim = 1)
        loss = loss_fn(y_logits, y)

        test_loss += loss.item()
        test_acc += (y_pred == y).sum().item() / len(y)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    print(f"Epoch {epoch + 1}/{n_epochs}|  Test Loss: {test_loss} | Test Accuracy: {test_acc }")

if __name__ == "__main__":
    train_fn()