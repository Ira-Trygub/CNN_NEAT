import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from population import Population
from yacs.config import CfgNode as CN
from customCNN import CustomCNN
from evolution import evaluate_genome, crossover_genomes, mutate_genome
import numpy as np
from test_acc import test_acc

cfg = CN(new_allowed = True)
cfg.defrost()
cfg.merge_from_file("cfg.yaml")


transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем изображения в тензоры
    transforms.Normalize((0.5,), (0.5,))  # Нормализуем данные (среднее и стандартное отклонение для канала)
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
example_image, _ = train_dataset[0]  # Получаем первый элемент из датасета
input_shape = example_image.shape 
print("input_shape: ", input_shape)
population_size = cfg.population_size
max_layers = cfg.max_layers 
num_generations = cfg.num_generations
population = Population(population_size, input_shape, min_layers=3, max_layers =10)
for generation in range(num_generations):
    fitness_scores = [evaluate_genome(genome, train_loader) for genome in population.population]
    sorted_genomes = [genome for _, genome in sorted(zip(fitness_scores, population.population))]
    best_genomes = sorted_genomes[:population_size // 2]
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = np.random.choise(best_genomes, 2 , replace = False)
        child = crossover_genomes(parent1, parent2)
        if np.random.rand() < 0.1:
            mutate_genome(child)
        new_population.append(child)
    population.population = new_population
    best_genome = population.population[0] 
    best_genome.print_layers()
    model = CustomCNN(best_genome)
    accuracy = test_acc( test_loader, model)
    
    




