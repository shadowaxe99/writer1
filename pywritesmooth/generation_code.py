import torch
from lstm_model import LSTMModel


def generate_svg(model_path, seed, sequence_length):
    model = LSTMModel(input_size=128, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sequence = [seed]
    for _ in range(sequence_length):
        input = torch.tensor(sequence[-1]).unsqueeze(0)
        output = model(input)
        sequence.append(output.detach().numpy())

    return sequence