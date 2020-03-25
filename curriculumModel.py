import torch
from torch import nn
import torch.nn.functional as F

class CurriculumNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=256):

        super(CurriculumNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_size)
        self.layer2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, teacher_input):
        return F.softmax(self.layer2(F.relu(self.layer1(teacher_input))), dim=-1)