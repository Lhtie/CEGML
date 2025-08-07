import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, device="cpu"):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x, lengths):
        out, _ = self.rnn(x)
        out = torch.stack([
            out[i, lengths[i], :] for i in range(x.shape[0])
        ])
        out = self.fc(out)
        return out