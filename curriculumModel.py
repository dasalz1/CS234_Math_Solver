import torch
from torch import nn
import torch.nn.functional as F

class CurriculumNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=256):

        super(CurriculumNetwork, self).__init__()
        hidden_layer_size = 128
        self.input_size = input_size
        self.output_size = output_size
        assert(input_size == output_size)
        # For curriculum network, input and output size should be same (category performance->category percentage)
        self.layer1 = nn.Linear(self.input_size, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, self.output_size)

    def forward(self, input):
        model = torch.nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.Softmax(dim=-1)
        )
        return model(input)
