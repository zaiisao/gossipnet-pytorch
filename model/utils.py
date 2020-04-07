import torch.nn as nn

def xavierInitialization(layer, biasValue=0.01):
    """
        Initialize the model with xavier initialization
        Input:
            layer: Could be single layer or a list of layers
    """
    if isinstance(layer, nn.Linear):
        # 'gain' is the scaling factor to be used
        # 'calculate_gain' returns the recommemded scaling factor
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(layer.bias, biasValue)