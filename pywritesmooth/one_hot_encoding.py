import numpy as np


def one_hot_encode(sequence, num_classes):
    encoding = np.zeros((len(sequence), num_classes))
    encoding[np.arange(len(sequence)), sequence] = 1
    return encoding