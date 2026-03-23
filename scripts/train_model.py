from analysis.analysis_loop import train_test_loop
from data.model_data import X_COLUMNS, Y_COLUMNS
from model.basicAnn import basicAnn


if __name__ == "__main__":
    training_file_name = "data_322_01_100"
    hidden_file_path = "model/hidden_test_data/hidden_test_data_2.h5"
    save_dir = "model/trained_models/"
    num_epochs = 500
    learning_rate = 0.001
    exclude_columns = ['u1_prev', 'u2_prev']
    # exclude_columns = []
    # exclude_columns = ['u1_prev', 'u2_prev', 'ee_dx_target', 'ee_dy_target']

    nn = basicAnn(input_size=len(X_COLUMNS) - len(exclude_columns), output_size=len(Y_COLUMNS))
    train_test_loop(training_file_name, hidden_file_path, save_dir, num_epochs, learning_rate, nn, exclude_columns)
