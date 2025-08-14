import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import json
from pathlib import Path

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "MNIST"
batch_size = 64
epochs = 15
lr = 0.001
out_dir = Path("./outputs")
out_dir.mkdir(exist_ok=True)

# Transformações
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset e DataLoader
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Modelo, Loss e Otimizador
model = SimpleCNN(in_channels=1, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Treino
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

# Salvar modelo
torch.save(model.state_dict(), out_dir / "best_model.pth")

# Salvar meta.json
meta = {
    "dataset": dataset_name,
    "classes": [str(i) for i in range(10)]
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f)

print(f"Treino finalizado! Modelos e meta.json salvos em {out_dir}")
