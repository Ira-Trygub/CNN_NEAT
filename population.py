import random
from genome import Genome

class Population:
    def __init__(self, population_size, input_shape, min_layers=3, max_layers=10):
        self.population_size = population_size
        self.input_shape = input_shape
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            genome = Genome(self.input_shape)
            num_layers = random.randint(self.min_layers, self.max_layers)
            
            # Добавление первого сверточного слоя
            conv1 = genome.add_layer('conv', {'filters': random.randint(16, 64), 'kernel_size': 3, 'stride': 1, 'padding': 1}, 'relu')
            
            prev_layer_index = conv1
            
            for layer_idx in range(1, num_layers - 1):
                layer_type = random.choice(['conv', 'pool', 'fc'])
                if layer_type == 'conv':
                    params = {'filters': random.randint(16, 128), 'kernel_size': random.randint(2, 5), 'stride': 1, 'padding': 1}
                elif layer_type == 'pool':
                    params = {'pool_type': 'max', 'kernel_size': random.randint(2, 3), 'stride': random.randint(1, 2)}
                elif layer_type == 'fc':
                    params = {'units': random.randint(64, 512)}
                new_layer_index = genome.add_layer(layer_type, params)
                genome.add_connection(prev_layer_index, new_layer_index)
                prev_layer_index = new_layer_index
            
            # Добавление последнего полносвязного слоя
            output_layer = genome.add_layer('fc', {'units': 10}, 'softmax')
            genome.add_connection(prev_layer_index, output_layer)
            
            # Добавление случайных дополнительных связей
            for _ in range(random.randint(0, num_layers)):
                in_layer = random.randint(0, len(genome.layers) - 2)
                out_layer = random.randint(in_layer + 1, len(genome.layers) - 1)
                genome.add_connection(in_layer, out_layer)
            
            population.append(genome)
        return population