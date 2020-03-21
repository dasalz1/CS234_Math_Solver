import torch
from torch import nn

class CurriculumNetwork(nn.Module):
    def __init__(self, input_output_size):
        super(CurriculumNetwork, self).__init__()
        hidden_layer_size = 128
        self.input_output_size = input_output_size
        self.layer1 = nn.Linear(self.input_output_size, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, self.input_output_size)

    def forward(self, input):
        model = torch.nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.Softmax(dim=-1)
        )
        return model(input)
