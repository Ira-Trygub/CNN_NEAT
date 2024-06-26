from genome import Genome, LayerGene
import numpy as np 
from customCNN import CustomCNN
import torch.optim as optim
import torch.nn as nn

def mutate_genome(genome):
    mutation_type = np.random.choice(['add_layer', 'remove_layer', 'modify_layer'])
    if mutation_type == 'add_layer':  # Добавить новый слой
        layer_type = np.random.choice(['conv', 'pool', 'fc'])
        if layer_type == 'conv':
            params = {'filters': np.random.randint(16, 128), 'kernel_size': np.random.randint(2, 5), 'stride': 1, 'padding': 1}
        elif layer_type == 'pool':
            params = {'pool_type': 'max', 'kernel_size': 2, 'stride': 2}
        elif layer_type == 'fc':
            params = {'units': np.random.randint(64, 512)}
        insert_position = np.random.randint(0, len(genome.layers))
        genome.layers.insert(insert_position, LayerGene(layer_type, params))
    elif mutation_type == 'remove_layer' and len(genome.layers) > 2:  # Удалить слой, если их больше двух
        remove_position = np.random.randint(0, len(genome.layers))
        genome.layers.pop(remove_position)
    elif mutation_type == 'modify_layer':  # Модификация существующего слоя
        layer_idx = np.random.randint(0, len(genome.layers))
        layer = genome.layers[layer_idx]
        if layer.layer_type == 'conv':
            layer.params['filters'] = np.random.randint(16, 128)
            layer.params['kernel_size'] = np.random.randint(2, 5)
        elif layer.layer_type == 'pool':
            layer.params['kernel_size'] = np.random.randint(2, 3)
        elif layer.layer_type == 'fc':
            layer.params['units'] = np.random.randint(64, 512)

def crossover_genomes(parent1, parent2):
    child = Genome(parent1.input_shape)
    for layer1, layer2 in zip(parent1.layers, parent2.layers):
        if np.random.rand() > 0.5:
            child.layers.append(layer1)
        else:
            child.layers.append(layer2)
    for conn1, conn2 in zip(parent1.connections, parent2.connections):
        if np.random.rand() > 0.5:
            child.connections.append(conn1)
        else:
            child.connections.append(conn2)
    return child



def evaluate_genome(genome, train_loader ):
    model = CustomCNN(genome)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

