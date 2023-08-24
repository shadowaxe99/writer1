import numpy as np
from xml.dom import minidom


class SVGData:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        doc = minidom.parse(self.file_path)
        path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
        doc.unlink()
        return path_strings

    def preprocess_data(self, data):
        processed_data = []
        for path_string in data:
            path_data = [float(i) for i in path_string.split() if i.replace('.', '', 1).isdigit()]
            processed_data.append(path_data)
        return np.array(processed_data)