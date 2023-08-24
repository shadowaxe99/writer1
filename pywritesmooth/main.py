import torch
from torch.utils.data import DataLoader
from lstm_model import LSTMModel
from LSTMTrainer import LSTMTrainer
from svg_data import SVGData


if __name__ == '__main__':
    # Load data
    svg_data = SVGData('path_to_svg_file')
    dataloader = DataLoader(svg_data, batch_size=32, shuffle=True)

    # Initialize model
    model = LSTMModel(input_size=128, hidden_size=256, num_layers=2)

    # Initialize trainer
    trainer = LSTMTrainer(model, learning_rate=0.001)

    # Train model
    for epoch in range(100):
        epoch_loss = trainer.train(dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

    # Save model
    torch.save(model.state_dict(), 'model.pth')