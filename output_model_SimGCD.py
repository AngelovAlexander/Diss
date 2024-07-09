import torch.nn as nn

class OutputSimGCD(nn.Module):
    def __init__(self, model):
        super(OutputSimGCD, self).__init__()
        self.model = model
        
    def forward(self, x):
        proj, logits = self.model(x)
        return logits

class OutputProjSimGCD(nn.Module):
    def __init__(self, model):
        super(OutputProjSimGCD, self).__init__()
        self.model = model
        
    def forward(self, x):
        proj, logits = self.model(x)
        return proj