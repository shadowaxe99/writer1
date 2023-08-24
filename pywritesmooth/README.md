# PyWriteSmooth

This project uses a Long Short-Term Memory (LSTM) model to generate smooth handwriting. The model is trained on SVG data and uses one-hot encoding for the input and output data. The handwriting smoothing logic is implemented in the `handwriting_smoothing.py` file. The LSTM model is implemented in the `lstm_model.py` file. The one-hot encoding is implemented in the `one_hot_encoding.py` file. The SVG data is loaded and preprocessed in the `svg_data.py` file. The training loop is implemented in the `training_loop.py` file. The generation code is implemented in the `generation_code.py` file. The testing is implemented in the `testing.py` file.

## Usage

1. Load and preprocess the SVG data.
2. Encode the data using one-hot encoding.
3. Create and train the LSTM model.
4. Generate new handwriting using the trained model.
5. Test the implementation.

## Requirements

- Python 3.7 or later
- PyTorch 1.7.1 or later
- NumPy 1.19.5 or later
- scikit-learn 0.24.1 or later
- xml 3.8 or later# py-write-smooth
