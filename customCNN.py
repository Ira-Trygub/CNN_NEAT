import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F 

class CustomCNN(nn.Module):
    def __init__(self, genome):
        super(CustomCNN, self).__init__()
        self.genome = genome
        self.layers = nn.ModuleList()
        self.build_model()
    
    def build_model(self):
        in_channels = self.genome.input_shape[0]
        flatten_added = False
        for layer_gene in self.genome.layers:
            if layer_gene.layer_type == 'conv':
                conv_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_gene.params['filters'],
                    kernel_size=layer_gene.params['kernel_size'],
                    stride=layer_gene.params['stride'],
                    padding=layer_gene.params['padding']
                )
                in_channels = layer_gene.params['filters']  #Amount of filters
                self.layers.append(conv_layer)
                flatten_added = False
            elif layer_gene.layer_type == 'pool':
                pool_layer = nn.MaxPool2d(
                    kernel_size=layer_gene.params['kernel_size'],
                    stride=layer_gene.params['stride']
                )
                self.layers.append(pool_layer)
                flatten_added = False
            elif layer_gene.layer_type == 'fc':
                #if isinstance(self.layers[-1], (nn.Conv2d, nn.MaxPool2d)):
                if not flatten_added:
                    self.layers.append(nn.Flatten())
                    in_features = self.calculate_flattened_size(self.genome.input_shape)
                    flatten_added = True
                else:
                    in_features = self.layers[-1].out_features
                fc_layer = nn.Linear(
                    in_features=in_features,
                    out_features=layer_gene.params['units']
                )
                in_features = layer_gene.params['units']
                self.layers.append(fc_layer)

    def calculate_flattened_size(self, input_shape):
        x = torch.rand(1, *input_shape)
        
        for layer in self.layers:

            x = layer(x)
            print(f"Shape after {layer}: {x.shape}")
            if x.numel() == 0:
                raise ValueError(f"Layer {layer} resulted in zero-sized tensor with input shape {x.shape}")
        return x.numel()
        
    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                x = F.relu(layer(outputs[-1]))
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(outputs[-1])
            elif isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            elif isinstance(layer, nn.Flatten):
                x = layer(outputs[-1])
            outputs.append(x)
        print(f"Shape after {layer}: {x.shape}") 
        for connection in self.genome.connections:
            if connection.enabled:
                outputs[connection.out_layer] += outputs[connection.in_layer]
        
        return outputs[-1]
    

