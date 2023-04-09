import itertools
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

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

class FoodSource:
    def __init__(self):
        self.position = np.random.randint(0, world_size[0]), np.random.randint(0, world_size[1])
        self.energy = 50

class Organism:
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

        if random.random() < mutation_probability:
            # Apply mutation

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

    def move(self):
        inputs = []
        for other in organisms:
            if other != self:
                distance = np.sqrt((self.position[0]-other.position[0])**2 + (self.position[1]-other.position[1])**2)
                inputs.append(1 / distance if distance != 0 else 0)

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

        # Update activations of the hidden neurons
        for i in range(2, self.num_neurons - 1):
            self.neurons[i].activate(inputs, self.connections)

        # Update activation of the output neurons
        for i in range(self.num_neurons - 1, self.num_neurons):
            self.neurons[i].activate([n.output for n in self.neurons[2:self.num_neurons - 1]], self.connections)

        # Calculate the velocity of the organism
        acceleration = np.array([self.neurons[-1].output, self.neurons[-2].output]) * 2 - 1
        self.velocity = (self.velocity + acceleration) / 2
        self.velocity = self.velocity / np.linalg.norm(self.velocity)
        self.position = np.array(self.position) + self.velocity
        self.position = np.mod(self.position, world_size)
        self.energy -= 1

        # Check if the organism is close enough to a food source and consume it
        food_eating_distance = 5
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

    def run(self):
        self.move()
        for other in organisms:
            if other != self:
                self.link(other)
        if np.random.random() < 0.1:
            if np.random.random() < 0.5:
                if len(self.memory) > 0:
                    index = np.random.randint(0, len(self.memory))
                    value = self.read_memory(index)
                    if value is not None:
                        self.transmit(value)
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

        #if np.random.random() < 0.01:
        #    if np.random.random() < 0.5:
       #         data = input()
      #          if data:
     #               self.receive(data)
    #        else:
   #             output_neurons = [neuron for neuron in self.neurons if type(neuron) == tuple and neuron[0] == "output"]
  #              if output_neurons:
 #                   neuron = output_neurons[np.random.randint(0, len(output_neurons))]
#                    value = chr(np.random.randint(32, 127))
 #                   neuron[1].receive(value)
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

#the update_plot function
def update_plot():
    print("Updating plot...")
    ax.clear()

    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_title("Organisms in the World")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    x = [organism.position[0] for organism in organisms]
    y = [organism.position[1] for organism in organisms]
    ax.scatter(x, y, c='b')

    x_food = [food.position[0] for food in food_sources]
    y_food = [food.position[1] for food in food_sources]
    ax.scatter(x_food, y_food, c='r', marker='x')

    plt.savefig(f'plot_{t}.png')
    plt.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
    print(f"Plot updated. Organisms alive: {len(organisms)}")  # Add this line


last_survivors = []

top_reproducers = []

import random



for t in range(10000):
    print(f"Iteration {t}...")
    for organism in organisms:
        organism.run()

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
    update_plot()


    if len(organisms) == 0:  # If all organisms died, spawn new ones with random neurons
        new_organisms = []
        
        # Include the last 6 survivors and reset their energy to 100
        for survivor in last_survivors:
            survivor.energy = 100
            new_organisms.append(survivor)
        
        # Spawn the remaining organisms randomly
        remaining_count = max(0, 100 - len(last_survivors))
        new_organisms.extend([Organism(mutation_probability) for _ in range(remaining_count)])
        
        organisms = new_organisms
        last_survivors = []



    if len(organisms) <= 6:
        last_survivors = list(organisms)

    if t % 100 == 0:
        update_plot()
        avg_energy = sum(o.energy for o in organisms) / len(organisms)
        print(f"Iteration {t}: Average energy = {avg_energy}")

        if top_reproducers:
            visualize_top_organism(top_reproducers[0], t)

    if len(organisms) <= 6:
        last_survivors = list(organisms)
    else:
        sorted_organisms = sorted(organisms, key=lambda o: o.reproduction_count, reverse=True)
        top_reproducers = sorted_organisms[:5]        


plt.show()


