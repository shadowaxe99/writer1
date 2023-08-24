import torch
from torch import nn, optim


class LSTMTrainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()
            output = self.model(batch['input'])
            loss = self.criterion(output, batch['target'])
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)