import numpy as np

def save_tensor_to_file(filename: str, tensor: np.ndarray):
    with open(filename, 'wb') as file:
        tensor.T.flatten().astype(np.float32).tofile(file)
