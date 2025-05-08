import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLSTM(nn.Module):
    """
    
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        self.hidden_layer1_size = 128
        self.hidden_layer2_size = 32
        self.stacked_lstm_layers = 2
        self.prob_dropout = 0.5

        self.layer1_lstm = nn.LSTM(
            input_size,
            self.hidden_layer1_size,
            num_layers=self.stacked_lstm_layers,
            dropout=0.2,
            batch_first=True
        )
        self.layer2_lstm = nn.LSTM(
            self.hidden_layer1_size,
            self.hidden_layer2_size,
            num_layers=self.stacked_lstm_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_layer2_size, output_size)

    def forward(self, x):
        lstm1_out, _ = self.layer1_lstm(x)
        lstm1_out = F.dropout(lstm1_out, p=self.prob_dropout, training=True)
        lstm2_out, _ = self.layer2_lstm(lstm1_out)
        lstm2_out = F.dropout(lstm2_out, p=self.prob_dropout, training=True)
        out = lstm2_out[:, -1, :]
        predictions = self.fc(out)
        return predictions[:,-1]


class StockLSTM(nn.Module):
    """ LSTM model for stock price prediction.
    This model is intended to use only Price (e.g. Close) as input and 
    consists of a preprocessing dense layer, an LSTM layer, and a final output layer.
    """

    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()

        # Dense Preprocessing Layer
        self.linear_1 = nn.Linear(in_features=input_size, out_features=hidden_layer_size) 
        self.relu = nn.ReLU()
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout) # Dropout layer; randomly setting activations to zero to prevent overfitting
        # Output Layer
        self.fc = nn.Linear(hidden_layer_size, output_size) 

    def forward(self, x):
        # Ensure input has shape (batch_size, sequence_length, input_size)
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer (no need for manual initialization)
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, time_steps, hidden_layer_size)

        # Take the last time step output
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_layer_size)

        # Apply dropout
        out = self.dropout(out)

        # Final prediction
        predictions = self.fc(out)

        return predictions[:,-1]
    

class MultiFeatureLSTM(nn.Module):
    """
    """
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()

        # Dense Preprocessing Layer
        self.linear_1 = nn.Linear(in_features=input_size, out_features=hidden_layer_size) 
        self.relu = nn.ReLU()
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)