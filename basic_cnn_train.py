import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)


class ToyImageDataset(Dataset):
    def __init__(self, size=800, image_size=16):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(1, self.image_size, self.image_size) * 0.2
        label = index % 2

        if label == 0:
            image[:, 3:7, 3:7] += 1.0
        else:
            image[:, 9:13, 9:13] += 1.0

        return image, torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 2),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = ToyImageDataset(size=800)
    test_data = ToyImageDataset(size=200)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_data)
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: loss={avg_loss:.4f}, test_acc={acc:.3f}")


if __name__ == "__main__":
    main()
