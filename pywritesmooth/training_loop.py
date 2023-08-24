from LSTMTrainer import LSTMTrainer
from torch.utils.data import DataLoader
from svg_data import SVGData


def train_model(model, epochs, learning_rate):
    trainer = LSTMTrainer(model, learning_rate)
    svg_data = SVGData('path_to_svg_file')
    dataloader = DataLoader(svg_data, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = trainer.train(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')