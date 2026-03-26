from analysis.analysis_loop import train_test_loop
from data.model_data import X_COLUMNS, Y_COLUMNS
from model.basicAnn import basicAnn
from model.lstm import CollisionAvoidanceLSTM

import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """
    Forces PyTorch, NumPy, and Python to be 100% deterministic.
    This ensures identical weight initialization, shuffling, and noise.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Force CuDNN to be deterministic (prevents GPU optimization randomness)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(42)
    
    training_file_name = "data_322_01_100"
    hidden_file_name = "hidden_test_data_2"
    # hidden_file_path = "model/hidden_test_data/hidden_test_data_2.h5"
    save_dir = "model/trained_models/"
    num_epochs = 200
    learning_rate = 0.001
    # exclude_columns = ['u1_prev', 'u2_prev']
    # exclude_columns = []
    exclude_columns = ['u1_prev', 'u2_prev', 'ee_dx_target', 'ee_dy_target']

    # nn = basicAnn(input_size=len(X_COLUMNS) - len(exclude_columns), output_size=len(Y_COLUMNS))
    from scripts.model_def import nn
    
    seq_length = 3
    train_test_loop(training_file_name, hidden_file_name, save_dir, num_epochs, learning_rate, nn, exclude_columns, seq_length)