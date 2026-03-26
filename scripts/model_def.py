from model.lstm import CollisionAvoidanceLSTM

nn = CollisionAvoidanceLSTM(
    input_size=10,
    hidden_size=32,   # try 32 or 64 instead of whatever you have now
    num_layers=1,     # start with 1
    output_size=2,
    dropout=0.2
)
