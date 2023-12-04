import torch
import torch.nn as nn

class ProjectionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):

        combined = torch.cat((x, hidden), 1)

        # Hidden state
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
