import torch
import h5py as h5
import numpy as np
import joblib

from model.basicAnn import basicAnn
from analysis.analysis_loop import run_model, get_hidden_data
from data.model_data import X_COLUMNS, Y_COLUMNS

def load_model(model, model_path, input_size, output_size):
    model.load_state_dict(torch.load(model_path))
    return model

def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler

