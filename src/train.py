import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.model import NeuralNetwork
from src.dataset import CO2Dataset, loadCSV

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

    data = CO2Dataset(loadCSV("data/emissions.csv"))
    batch_size = 24
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 50

    while num_epochs > 0:
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {20-num_epochs+1}/{20}, Loss: {loss.item():.6f}')
        num_epochs -= 1
