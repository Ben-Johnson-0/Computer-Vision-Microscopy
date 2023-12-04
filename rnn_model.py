import torch
import torch.nn as nn

class ProjectionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ProjectionRNN, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [Batch, Sequence_length, input_size]
        # lstm_out shape: [Batch, sequence_length, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Last time step's out is the projection 
        # projection shape: [Batch, hidden_size]
        projection = lstm_out[:, -1, :]

        # Project hidden state to output size
        # output shape: [Batch, output_size]
        output = self.linear(projection)
        # output = self.softmax(output)

        return output
