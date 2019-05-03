import numpy
import pickle
import GA
import ANN
from DataPreparation import DataPreparation_GA
import matplotlib.pyplot

dp = DataPreparation_GA()
filename = '../nnga/data.txt'
dataset = dp.load_txt(filename)
for i in range(len(dataset[0]) - 1):
    dp.str_column_to_float(dataset, i)
    # convert class column to integers
dp.str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
minmax = dp.dataset_minmax(dataset)
dp.normalize_dataset_classification(dataset, minmax) 
data_inputs = [row[0:4] for row in dataset]
data_outputs = [row[-1] for row in dataset]

data_inputs = numpy.array(data_inputs)
data_outputs= numpy.array(data_outputs)

sol_per_pop = 8
num_parents_mating = 4
num_generations = 400
mutation_percent = 10

#Creating the initial population.
initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
    HL1_neurons = 150
    input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, 

                                             size=(data_inputs.shape[1], HL1_neurons))

    HL2_neurons = 60
    HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, 

                                             size=(HL1_neurons, HL2_neurons))

    output_neurons = 2
    HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, 

                                              size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(numpy.array([input_HL1_weights, 

                                                HL1_HL2_weights, 

                                                HL2_output_weights]))

pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = GA.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = numpy.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = GA.vector_to_mat(pop_weights_vector, 
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(pop_weights_mat, 
                          data_inputs, 
                          data_outputs, 
                          activation="sigmoid")

    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(pop_weights_vector, 

                                    fitness.copy(), 

                                    num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents,

                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover, 

                                     mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
print("Accuracy of the best solution is : ", acc)

matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=15)

f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()