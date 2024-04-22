from torch import nn
from torch import optim
from torchvision import torch
from src.model import NeuralNetwork
from src.dataset import train_dataloader, test_dataloader_emnist, test_dataloader_semeion

def train():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.025)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 50

    while num_epochs > 0:
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {20-num_epochs+1}/{20}, Loss: {loss.item():.6f}')
        num_epochs -= 1
