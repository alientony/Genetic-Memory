import itertools
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import time

def visualize_top_organism(organism, iteration):
    G = nx.DiGraph()
    
    for i, neuron in enumerate(organism.neurons):
        G.add_node(i, label=f'N{i}')
        
    # Connect nodes based on neuron connections
    for i, neuron in enumerate(organism.neurons):
        for j, other_neuron in enumerate(organism.neurons):
            if neuron != other_neuron:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color='orange', with_labels=True, node_size=1500)
    plt.title(f"Top Organism's Structure at Iteration {iteration}")

    # Add the memory bank content to the bottom of the image
    memory_content = organism.memory
    plt.annotate(f"Memory: {memory_content}", xy=(0, 0), xycoords='axes fraction', fontsize=12, ha='left', va='bottom')

    plt.savefig(f"top_organism_structure_{iteration}.png")
    plt.close()


# Define the size of the world
world_size = (1000, 1000)

# Define the class for neurons
class Neuron:
    def __init__(self):
        self.activation = 0
        self.output = 0

    def activate(self, inputs, connections):
        weighted_inputs = [connections.get((i, self), 0) * input for i, input in enumerate(inputs)]
        self.activation = sum(weighted_inputs)
        self.output = 1 / (1 + np.exp(-self.activation))

FOOD_MIN_ENERGY = 10
FOOD_MAX_ENERGY = 100


class FoodSource:
    def __init__(self):
        self.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])
        self.energy = 50
def create_food_sources(num_food_sources, world):
    food_sources = []
    for _ in range(num_food_sources):
        x = random.randint(0, world.shape[1] - 1)
        y = random.randint(0, world.shape[0] - 1)
        food_source = FoodSource()
        food_source.position = (x, y)
        food_source.energy = random.uniform(FOOD_MIN_ENERGY, FOOD_MAX_ENERGY)
        food_sources.append(food_source)
    return food_sources



class Organism:
    def __init__(self, world):
        self.world = world
        self.x = None
        self.y = None
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability
        self.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])
        self.neurons = []
        self.num_neurons = 11  # Increase the number of neurons to 11
        for i in range(self.num_neurons):
            neuron = Neuron()
            neuron.activation = np.random.randn()  # Initialize neuron activation with random values
            self.neurons.append(neuron)
        self.connections = {(i, j): np.random.randn() for i in range(self.num_neurons) for j in range(self.num_neurons) if i != j and np.random.random() < 0.5}
        self.memory = ""
        self.energy = 100
        self.velocity = np.array([0, 0])
        self.reproduction_count = 0

    def reproduce(self, mutation_probability):
        offspring = Organism(mutation_probability)  # Create a new instance of Organism
        offspring.connections = self.connections.copy()  # Copy connections from the parent to the offspring

        # Copy neurons from the parent to the offspring, with a chance of mutation
        for i in range(self.num_neurons):
            offspring.neurons[i].activation = self.neurons[i].activation
            if random.random() < 0.05:  # 5% mutation rate
                offspring.neurons[i].activation += np.random.normal(0, 0.1)  # Add random noise to the activation

        # Mutate connections with a chance of mutation
        for key in offspring.connections:
            if random.random() < self.mutation_probability:
                offspring.connections[key] += np.random.normal(0, 0.1)  # Add random noise to the connection weight

        # Position the offspring close to the parent
        offspring.position = tuple(np.random.randint(max(0, self.position[i] - 10), min(world_size[i], self.position[i] + 10)) for i in range(2))
        offspring.energy = self.energy / 2  # Transfer half of the parent's energy to the offspring
        self.energy /= 2  # The parent loses half of its energy during reproduction
        self.reproduction_count += 1

        return offspring

    def update(self, food_sources):
        inputs = [food_source.energy / (np.linalg.norm(np.array(self.position) - np.array(food_source.position)) + 1e-8) for food_source in food_sources]

        # Add memory neuron value to the inputs
        inputs.append(self.neurons[10].output)

        for neuron in self.neurons:
            neuron.activate(inputs, self.connections)

        self.velocity = np.array([self.neurons[8].output, self.neurons[9].output]) * 2 - 1
        self.position = tuple(np.mod(np.array(self.position) + self.velocity, world_size))

        # Update memory content
        memory_char = chr(int(self.neurons[7].output * 26) + 65)
        self.memory += memory_char
        if len(self.memory) > 5:
            self.memory = self.memory[-5:]

        self.energy -= 0.1  # Every update the organism loses a small amount of energy

# Simulation parameters
num_iterations = 5000
num_organisms = 50
mutation_probability = 0.05
initial_energy = 100
food_sources = [FoodSource() for _ in range(10)]
organisms = [Organism(mutation_probability) for _ in range(num_organisms)]

NUM_FOOD_SOURCES = 20


NUM_ORGANISMS = 100  # Number of organisms to start with

if __name__ == "__main__":
    world = np.zeros((50, 50))

    # Create and add organisms to the world
    organisms = []
    for _ in range(NUM_ORGANISMS):
        x = random.randint(0, world.shape[1] - 1)
        y = random.randint(0, world.shape[0] - 1)
        energy = random.randint(1, 100)
        organism = Organism(world)
        organism.x = x
        organism.y = y
        organism.energy = energy
        world[y, x] = 1  # Mark the organism's position in the world
        organisms.append(organism)

    # ... (Continue with the main simulation loop)

    food_sources = create_food_sources(NUM_FOOD_SOURCES, world)
    NUM_ITERATIONS = 1000
    for _ in range(NUM_ITERATIONS):
        # Update organisms and world
        for organism in organisms:
            organism.update(food_sources)

        # Refresh the world and the organisms' energy
        world, food_sources = refresh_world(world, organisms, food_sources)
    # Main simulation loop
    for iteration in range(1000):
        # Update organisms
        for organism in organisms:
            organism.update()

        # Remove dead organisms
        organisms = [organism for organism in organisms if organism.energy > 0]

        # Reproduction
        new_organisms = []
        for organism in organisms:
            offspring = organism.reproduce()
            if offspring:
                new_organisms.append(offspring)

        organisms.extend(new_organisms)

        # Check if there are any organisms left in the simulation
        if not organisms:
            print("No organisms left in the simulation.")
            break

        # Find the organism with the highest energy
        top_organism = max(organisms, key=lambda x: x.energy)
        print(f"Iteration {iteration + 1}: Top organism at ({top_organism.x}, {top_organism.y}) with energy {top_organism.energy}")


food_sources = create_food_sources(NUM_FOOD_SOURCES, world)



for iteration in range(num_iterations):
    # Update organisms and handle reproduction
    new_organisms = []
    for organism in organisms:
        organism.update(food_sources)
        if organism.energy >= initial_energy * 2:
            offspring = organism.reproduce(mutation_probability)
            new_organisms.append(offspring)
    organisms.extend(new_organisms)

    # Organisms consume food and regain energy
    for organism in organisms:
        for food_source in food_sources:
            if np.linalg.norm(np.array(organism.position) - np.array(food_source.position)) < 10:
                organism.energy += food_source.energy
                food_source.energy = 0

    # Refresh food sources
    for food_source in food_sources:
        if food_source.energy == 0:
            food_source.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])
            food_source.energy = 50

    # Remove organisms with energy below 1
    organisms = [organism for organism in organisms if organism.energy > 1]

    # Visualize the top organism every 50 iterations
    if iteration % 50 == 0:
        if organisms:  # Check if the organisms list is not empty
            top_organism = max(organisms, key=lambda x: x.energy)
            visualize_top_organism(top_organism, iteration)
        else:
            print("No organisms left")
            break

    print(f"Iteration {iteration}: {len(organisms)} organisms")
