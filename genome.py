class LayerGene:
    def __init__(self, layer_type, params, activation='relu'):
        self.layer_type = layer_type  # 'conv', 'pool', 'fc'
        self.params = params  # Параметры слоя, зависят от типа
        self.activation = activation  # Активационная функция для слоя

class ConnectionGene:
    def __init__(self, in_layer, out_layer, enabled=True):
        self.in_layer = in_layer  # Индекс входного слоя
        self.out_layer = out_layer  # Индекс выходного слоя
        self.enabled = enabled  # Включена ли связь

class Genome:
    def __init__(self, input_shape):
        self.layers = []
        self.connections = []
        self.input_shape = input_shape
        
    
    def add_layer(self, layer_type, params, activation='relu'):
        new_layer = LayerGene(layer_type, params, activation)
        self.layers.append(new_layer)
        return len(self.layers) - 1  # Возвращаем индекс нового слоя
    
    def add_connection(self, in_layer, out_layer):
        new_connection = ConnectionGene(in_layer, out_layer)
        self.connections.append(new_connection)

    def print_layers(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Type={layer.layer_type}, Params={layer.params}, Activation={layer.activation}")