import unittest
from lstm_model import LSTMModel
from LSTMTrainer import LSTMTrainer


class TestLSTM(unittest.TestCase):
    def test_lstm_model(self):
        model = LSTMModel(128, 256, 2)
        self.assertIsInstance(model, LSTMModel)

    def test_lstm_trainer(self):
        model = LSTMModel(128, 256, 2)
        trainer = LSTMTrainer(model, 0.001)
        self.assertIsInstance(trainer, LSTMTrainer)


if __name__ == '__main__':
    unittest.main()