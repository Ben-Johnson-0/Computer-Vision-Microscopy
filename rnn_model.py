import torch
import torch.nn as nn

class ProjectionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ProjectionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # x shape: [Batch, Sequence_length, input_size]
        # lstm_out shape: [Batch, sequence_length, hidden_size]
        lstm_out, (final_hidden, final_cell) = self.lstm(x, (h_0, c_0))

        # Last time step out is the projection 
        # projection shape: [Batch, hidden_size]
        projection = final_hidden[-1]

        # Project hidden state to output size
        # output shape: [Batch, output_size]
        output = self.linear(projection)
        # output = self.softmax(output)

        return output
