import itertools
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from itertools import cycle








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

from enum import Enum

class NeuronType(Enum):
    MEMORY_READ = 1
    MEMORY_WRITE = 2
    MIDDLE = 3
    OBJECT_SENSING = 4
    DISTANCE_SENSING = 5
    INCREASE_VELOCITY = 6
    TURN_LEFT = 7
    TURN_RIGHT = 8



# Define the class for neurons
class Neuron:
    def __init__(self, neuron_type):
        self.neuron_type = neuron_type
        self.activation = 0
        self.output = 0

    def activate(self, inputs, connections):
        weighted_inputs = [connections.get((i, self), 0) * input for i, input in enumerate(inputs)]
        self.activation = sum(weighted_inputs)
        self.output = 1 / (1 + np.exp(-self.activation))


class FoodSource:
    def __init__(self):
        self.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])
        self.energy = 100


class Organism:
    def __init__(self, mutation_probability, reward_for_sharing=5, initial_neurons=9, initial_connections=9, num_neurons=None):

        self.brain = self.create_random_brain()

        self.reward_for_sharing = reward_for_sharing
        self.mutation_probability = mutation_probability
        self.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])

        if num_neurons is None:
            self.num_neurons = initial_neurons + 3  # Add 3 new neurons for movement control
        else:
            self.num_neurons = num_neurons

        neuron_types = list(NeuronType)
        random.shuffle(neuron_types)
        neuron_type_cycle = cycle(neuron_types)

        self.neurons = [Neuron(next(neuron_type_cycle)) for _ in range(self.num_neurons)]

        # Initialize connections based on neuron types
        self.connections = {}
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    src_neuron_type = self.neurons[i].neuron_type
                    dest_neuron_type = self.neurons[j].neuron_type
                    if (
                        src_neuron_type == NeuronType.MEMORY_READ
                        and dest_neuron_type in [NeuronType.MIDDLE, NeuronType.MEMORY_WRITE]
                    ) or (
                        src_neuron_type == NeuronType.MEMORY_WRITE
                        and dest_neuron_type in [NeuronType.MIDDLE, NeuronType.MEMORY_READ]
                    ) or (
                        src_neuron_type == NeuronType.OBJECT_SENSING
                        and dest_neuron_type == NeuronType.MIDDLE
                    ) or (
                        src_neuron_type == NeuronType.DISTANCE_SENSING
                        and dest_neuron_type == NeuronType.MIDDLE
                    ) or (
                        src_neuron_type == NeuronType.MIDDLE
                        and dest_neuron_type in neuron_types
                    ):
                        self.connections[(i, j)] = np.random.randn() * 0.1

        self.memory = ""
        self.energy = 150
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.reproduction_count = 0

    def create_random_brain(self):
        # Define the architecture of the neural network
        # Here's a simple example with one hidden layer
        input_size = 11
        hidden_size = 20
        output_size = 5

        # Initialize weights and biases with random values
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(output_size)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def act(self, state):
        # Convert the state to a numpy array
        state = np.array(state)

        # Perform the forward pass of the neural network
        Z1 = np.dot(state, self.brain['W1']) + self.brain['b1']
        A1 = np.tanh(Z1)  # Apply activation function (tanh) to the hidden layer
        Z2 = np.dot(A1, self.brain['W2']) + self.brain['b2']
        A2 = np.tanh(Z2)  # Apply activation function (tanh) to the output layer

        # Choose the action with the highest output value
        action = np.argmax(A2)

        return action





    def reproduce(self, mutation_probability):
        offspring = Organism(mutation_probability, num_neurons=self.num_neurons)  # Create a new instance of Organism

        offspring.connections = self.connections.copy()  # Copy connections from the parent to the offspring
        neuron_types = list(NeuronType)

        # Copy neurons from the parent to the offspring, with a chance of mutation
        for i in range(self.num_neurons):
            offspring.neurons[i].activation = np.random.uniform(-1, 1)
            if random.random() < 0.05:  # 5% mutation rate
                offspring.neurons[i].activation += np.random.normal(0, 0.1)  # Add random noise to the activation

        if random.random() < mutation_probability:
            # Apply mutation

                # Add a new neuron with a certain probability (e.g., 5%)
            if random.random() < 0.05:
                new_neuron = Neuron(random.choice(neuron_types))
                new_neuron.activation = np.random.randn()
                offspring.neurons.append(new_neuron)
                offspring.num_neurons += 1
                        # Add connections between the new neuron and existing neurons
                for i in range(offspring.num_neurons - 1):
                    if random.random() < 0.5:
                        offspring.connections[(i, offspring.num_neurons - 1)] = np.random.randn()
                    if random.random() < 0.5:
                            offspring.connections[(offspring.num_neurons - 1, i)] = np.random.randn()

            # Remove a neuron with a certain probability (e.g., 5%)
            if random.random() < 0.05 and offspring.num_neurons > 1:
                removed_neuron_index = random.randrange(offspring.num_neurons)
                del offspring.neurons[removed_neuron_index]
                offspring.num_neurons -= 1

                # Remove connections associated with the removed neuron
                for i, j in itertools.product(range(offspring.num_neurons + 1), repeat=2):
                    if (i == removed_neuron_index) or (j == removed_neuron_index):
                        if (i, j) in offspring.connections:
                            del offspring.connections[i, j]



            # Mutate connections and weights
            offspring.connections = self.connections.copy()
            for i, j in itertools.product(range(offspring.num_neurons), repeat=2):
                if i != j:
                    if (i, j) not in offspring.connections:
                        if np.random.random() < 0.05:  # 5% chance of adding a connection
                            offspring.connections[i, j] = np.random.randn()
                    else:
                        if np.random.random() < 0.05:  # 5% chance of removing a connection
                            del offspring.connections[i, j]
                        elif np.random.random() < 0.05:  # 5% chance of mutating the weight of a connection
                            offspring.connections[i, j] += np.random.normal(0, 0.1)

        offspring.position = self.position
        offspring.energy = self.energy / 2
        self.energy /= 2
        return offspring


    def act(self, state):
        # Convert the state to a numpy array
        state = np.array(state)

        # Implement the forward pass of the neural network using the state as input
        hidden = np.maximum(0, np.dot(state, self.brain['W1']) + self.brain['b1'])
        output = np.dot(hidden, self.brain['W2']) + self.brain['b2']

        # Softmax activation for the output layer
        output_exp = np.exp(output - np.max(output))
        output_softmax = output_exp / np.sum(output_exp)

        return output_softmax
    


    def move(self):
        organism_positions = np.array([organism.position for organism in organisms if organism != self])
        
        if organism_positions.size == 0:
            return
        
        distances = np.sqrt(np.sum((organism_positions - self.position) ** 2, axis=1))

        closest_indices = np.argsort(distances)[:9]
        closest_distances = distances[closest_indices]

        inputs = closest_distances.tolist()

        # Pad the inputs with zeros when there are fewer than 9 other organisms
        while len(inputs) < 9:
            inputs.append(0)

        # Add a food sensing neuron input
        food_sensing_radius = 2  # 2-foot radius for sensing food
        food_sensed = 0
        for food_source in food_sources:
            distance_to_food = np.sqrt((self.position[0] - food_source.position[0]) ** 2 + (self.position[1] - food_source.position[1]) ** 2)
            if distance_to_food <= food_sensing_radius:
                food_sensed = 1
                break
        inputs.append(food_sensed)

        # Add sight neuron input
        sight_angle = 30  # Angle in degrees for the sight range
        sight_distance = 50  # Maximum distance for the sight range
        sight_detected = 0
        for other in organisms:
            if other != self:
                dx = other.position[0] - self.position[0]
                dy = other.position[1] - self.position[1]
                angle = np.arctan2(dy, dx) - np.arctan2(self.velocity[1], self.velocity[0])
                angle = np.rad2deg((angle + np.pi) % (2 * np.pi) - np.pi)
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if -sight_angle / 2 <= angle <= sight_angle / 2 and distance <= sight_distance:
                    sight_detected = 1
                    break
        inputs.append(sight_detected)

        # Calculate the velocity of the organism
        action = self.act(inputs)
        increase_velocity = action[0]
        turn_left = action[1]
        turn_right = action[2]

        # ... (the rest of the code for hidden neurons and movement control neurons)

        acceleration = np.array([increase_velocity, 0])
        self.velocity = (self.velocity + acceleration) / 2
        self.velocity = self.velocity / (np.linalg.norm(self.velocity) + 1e-10)

        # Calculate the turning angle
        turning_angle = turn_left - turn_right
        rotation_matrix = np.array([[np.cos(turning_angle), -np.sin(turning_angle)],
                                    [np.sin(turning_angle), np.cos(turning_angle)]])
        self.velocity = np.matmul(rotation_matrix, self.velocity)

        self.position = np.array(self.position) + self.velocity
        self.position = np.mod(self.position, world_size)
        self.energy -= 1

        # Check if the organism is close enough to a food source and consume it
        food_eating_distance = 3  # 3-foot distance for eating food
        for food_source in food_sources:
            distance_to_food = np.sqrt((self.position[0] - food_source.position[0]) ** 2 + (self.position[1] - food_source.position[1]) ** 2)
            if distance_to_food <= food_eating_distance:
                self.energy += food_source.energy
                food_sources.remove(food_source)
                food_sources.append(FoodSource())




    def link(self, other):
        distance = np.sqrt((self.position[0]-other.position[0])**2 + (self.position[1]-other.position[1])**2)
        if distance == 1:
            self.neurons.append(other)
            other.neurons.append(self)
            self.energy -= 5
            other.energy -= 5

    def read_memory(self, index):
        if index >= 0 and index < len(self.memory):
            return self.memory[index]
        else:
            return None

    def write_memory(self, index, value):
        if index >= 0:
            if index < len(self.memory):
                self.memory = self.memory[:index] + value + self.memory[index+1:]
            else:
                self.memory += value
            self.energy -= 2

    def receive(self, data):
        if type(data) == str:
            self.memory += data
            self.energy -= 1


    def transmit(self, data):
        for neuron in self.neurons:
            if type(neuron) == tuple and neuron[0] == "transmitting":
                neuron[1].receive(data)
                self.energy -= 1
                self.energy += self.reward_for_sharing  # Reward for sharing information

    def share_information(self):
        if len(self.memory) > 0:
            index = np.random.randint(0, len(self.memory))
            value = self.read_memory(index)
            if value is not None:
                self.transmit(value)

    def run(self):
        self.move()
        for other in organisms:
            if other != self:
                self.link(other)
        if np.random.random() < 0.1:
            if np.random.random() < 0.5:
                self.share_information()
            else:
                index = np.random.randint(0, len(self.memory)+1)
                value = chr(np.random.randint(32, 127))
                self.write_memory(index, value)

        for neuron in self.neurons:
            if type(neuron) == tuple and neuron[0] == "receiving":
                receiving_neighbors = [n for n in self.neurons if type(n) == tuple and n[0] == "receiving" and n != neuron]
                if receiving_neighbors:
                    self.energy -= 10
            elif type(neuron) == tuple and neuron[0] == "transmitting":
                transmitting_neighbors = [n for n in self.neurons if type(n) == tuple and n[0] == "transmitting" and n != neuron]
                if transmitting_neighbors:
                    self.energy -= 10
        if np.random.random() < 0.1:
            self.energy += 1 if np.random.random() < 0.5 else -1
            self.energy = max(0, self.energy)
    
        # Check if the organism's energy is below 0, and remove it from the organisms list if it dies
        if self.energy <= 0:
            organisms.remove(self)


        for neuron in self.neurons:
            if type(neuron) == tuple and neuron[0] == "receiving":
                receiving_neighbors = [n for n in self.neurons if type(n) == tuple and n[0] == "receiving" and n != neuron]
                if receiving_neighbors:
                    self.energy -= 10
            elif type(neuron) == tuple and neuron[0] == "transmitting":
                transmitting_neighbors = [n for n in self.neurons if type(n) == tuple and n[0] == "transmitting" and n != neuron]
                if transmitting_neighbors:
                    self.energy -= 10
        if np.random.random() < 0.1:
            self.energy += 1 if np.random.random() < 0.5 else -1
            self.energy = max(0, self.energy)

mutation_probability = 0.05  # Replace this with your desired value
organism = Organism(mutation_probability)
offspring = organism.reproduce(mutation_probability=0.2)  # mutation_probability will be 0.2


# Initialization of organisms and food sources
organisms = [Organism(mutation_probability) for _ in range(100)]
food_sources = [FoodSource() for _ in range(100)]

#set up the plot
fig, ax = plt.subplots()
plt.ion()  # Add this line
ax.set_xlim(0, world_size[0])
ax.set_ylim(0, world_size[1])
ax.set_title("Organisms in the World")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

reproduction_threshold = 200  # Adjust this value to control how much energy is required for reproduction


last_survivors = []

top_reproducers = []

import random

import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the display
world_size_pixels = (1000, 1000)  # Adjust the size of the window as needed
screen = pygame.display.set_mode(world_size_pixels)
pygame.display.set_caption('Organisms World')
max_organisms = 200  # Set the maximum number of organisms allowed in the simulation


def update_display():
    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw organisms
    for organism in organisms:
        pygame.draw.circle(screen, (0, 0, 255), (int(organism.position[0] * world_size_pixels[0] / world_size[0]), int(organism.position[1] * world_size_pixels[1] / world_size[1])), 5)

    # Draw food sources
    for food_source in food_sources:
        pygame.draw.circle(screen, (0, 255, 0), (int(food_source.position[0] * world_size_pixels[0] / world_size[0]), int(food_source.position[1] * world_size_pixels[1] / world_size[1])), 5)

    # Update the display
    pygame.display.flip()


for t in range(10000):
    print(f"Iteration {t}...")
    
    # Handle events and update the display
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    update_display()
    pygame.time.delay(50)  # Adjust the delay as needed

    for organism in organisms:
        organism.run()




    # Check if the organism has enough energy to reproduce and if the total number of organisms is less than the maximum allowed
    if organism.energy >= reproduction_threshold and len(organisms) < max_organisms:
        offspring = organism.reproduce(mutation_probability)
        organisms.append(offspring)



    if len(organisms) == 0:  # If all organisms died, spawn new ones with random neurons
        new_organisms = []

        # Include the last 6 survivors and reset their energy to 100
        for survivor in last_survivors:
            survivor.energy = 100
            new_organisms.append(survivor)

        # Include the top 5 reproducers and reset their energy to 100
        for reproducer in top_reproducers:
            reproducer.energy = 100
            new_organisms.append(reproducer)

        # Spawn the remaining organisms randomly
        remaining_count = max(0, 100 - len(last_survivors) - len(top_reproducers))
        new_organisms.extend([Organism(mutation_probability) for _ in range(remaining_count)])

        organisms = new_organisms
        last_survivors = []
        top_reproducers = []


    if len(organisms) <= 6:
        last_survivors = list(organisms)

    if t % 50 == 0:
        # update_plot()
        avg_energy = sum(o.energy for o in organisms) / len(organisms)
        print(f"Iteration {t}: Average energy = {avg_energy}")

        if top_reproducers:
            visualize_top_organism(top_reproducers[0], t)

    if len(organisms) <= 6:
        last_survivors = list(organisms)
    else:
        sorted_organisms = sorted(organisms, key=lambda o: o.reproduction_count, reverse=True)
        top_reproducers = sorted_organisms[:5]        

# update_plot()
plt.show()


