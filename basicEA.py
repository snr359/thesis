
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import random
import csv
import datetime
import os
import configparser
import shutil

import fitness_functions as ff


class popi:
    def __init__(self):
        self.genotype = None
        self.fitness = None
    def initializeRandom(self, dimensionality, initializationRange):
        self.genotype = list(random.uniform(-0.5, 0.5)*initializationRange for _ in range(dimensionality))

def mean(l):
    lList = list(l)
    return float(sum(lList)) / len(lList)

def kTournamentSelection(pop, k):
    tournament = random.sample(pop, k)
    winner = max(tournament, key=lambda x: x.fitness)
    return winner

def fitPropSelection(pop):
    minFitness = min(list(p.fitness for p in pop))
    totalFitness = sum(p.fitness-minFitness for p in pop)

    r = random.uniform(0, totalFitness)
    for i in range(len(pop)):
        if r < (pop[i].fitness - minFitness):
            return pop[i]
        r -= (pop[i].fitness - minFitness)

    print("WARNING: overrun fitness proportional fitness list with {0} remaining".format(r))

def recombinePop(parent1, parent2):
    newChild = popi()
    geneLength = len(parent1.genotype)
    newChild.genotype = np.zeros(geneLength)
    for i in range(geneLength):
        if random.random() > 0.5:
            newChild.genotype[i] = parent1.genotype[i]
        else:
            newChild.genotype[i] = parent2.genotype[i]
    return newChild

def mutatePop(child):
    geneLength = len(child.genotype)
    for i in range(geneLength):
        child.genotype[i] += random.triangular(-1,1, 0)

def evaluatePop(child, fitnessFunction):
    fitness = ff.get_fitness(child.genotype, fitnessFunction)
    child.fitness = fitness

def logResults(logPath, averageFitness, bestFitness):
    # outputs an evolution result log in csv format
    evalNums = sorted(list(averageFitness.keys()))
    with open(logPath) as f:
        writer = csv.writer(f)
        writer.writerow(evalNums)
        writer.writerow(averageFitness[e] for e in evalNums)
        writer.writerow(bestFitness[e] for e in evalNums)

def basicEA(resultsPath):
    # read the fitness function from the configuration file
    fitnessFunction= config['experiment']['fitness function']

    # Read other parameters
    parentTournamentK = config.getint('EA','parent tournament k value')
    survivalTournamentK = config.getint('EA','survival tournament k value')
    mutationRate = config.getfloat('EA','mutation rate')
    popSize = config.getint('EA', 'mu')
    childSize = config.getint('EA','lambda')
    maxEvals = config.getint('EA','maximum fitness evaluations')

    parentSelection = config.get('EA','parent selection algorithm').lower().replace(' ', '').replace('-', '')
    survivalSelection = config.get('EA','survival selection algorithm').lower().replace(' ', '').replace('-', '')

    numRuns = config.getint('EA','runs')

    dim = config.getint('experiment', 'dimensionality')

    initializationRange = config.getint('EA','initialization range')


    allRunsAverageFitnesses = []
    allRunsBestFitnesses = []

    with open(resultsPath + "/EAparameters.log", 'w') as EAparamsFile:
        for param, val in zip(["parentTournamentK", "survivorTournamentK", "parentSelection", "survivalSelection", "mutationRate", "popSize", "childSize", "maxEvals"],
                              [parentTournamentK, survivalTournamentK, parentSelection, survivalSelection, mutationRate, popSize, childSize, maxEvals]):
            EAparamsFile.write(param + ": " + str(val) + '\n')


    resultFile = open(resultsPath + "/results0.csv", 'w')
    resultWriter = csv.writer(resultFile)

    for r in range(numRuns):
        print('Run {0}'.format(r))
        averageFitnessDict = dict()
        bestFitnessDict = dict()

        # initialization
        pop = []
        for i in range(popSize):
            newIndividual = popi()
            newIndividual.initializeRandom(dim, initializationRange)
            evaluatePop(newIndividual, fitnessFunction)
            pop.append(newIndividual)
        evals = popSize

        avgFitness = sum(p.fitness for p in pop) / len(pop)
        bestFitness = max(p.fitness for p in pop)
        # print("Average fitness")
        # print(avgFitness)
        # print("Best Fitness")
        # print(bestFitness)

        averageFitnessDict[evals] = avgFitness
        bestFitnessDict[evals] = bestFitness

        while evals < maxEvals:
            # parent selection/recombination
            children = []
            while len(children) < childSize:
                if parentSelection == "ktournament":
                    parent1 = kTournamentSelection(pop, parentTournamentK)
                    parent2 = kTournamentSelection(pop, parentTournamentK)
                elif parentSelection == 'fitnessproportional':
                    parent1 = fitPropSelection(pop)
                    parent2 = fitPropSelection(pop)
                elif parentSelection == 'truncation':
                    pop.sort(key = lambda p: p.fitness, reverse=True)
                    index = len(children) % len(pop)
                    parent1 = pop[index]
                    parent2 = pop[index+1]
                else:
                    print('WARNING: parent selection method {0} not recognized. Defaulting to fitness proportional'.format(parentSelection))
                    parent1 = fitPropSelection(pop)
                    parent2 = fitPropSelection(pop)
                newChild = recombinePop(parent1, parent2)
                if random.random() < mutationRate:
                    mutatePop(newChild)
                evaluatePop(newChild, fitnessFunction)
                evals += 1
                children.append(newChild)

            # population merging
            pop += children

            # log stats
            avgFitness = sum(p.fitness for p in pop) / len(pop)
            bestFitness = max(p.fitness for p in pop)
            # print("Average fitness")
            # print(avgFitness)
            # print("Best Fitness")
            # print(bestFitness)
            averageFitnessDict[evals] = avgFitness
            bestFitnessDict[evals] = bestFitness


            # survival selection
            if survivalSelection == 'ktournament':
                survivors = []
                while len(survivors) < popSize:
                    survivor = kTournamentSelection(pop, survivalTournamentK)
                    survivors.append(survivor)
                    pop.remove(survivor)
                pop = survivors
            elif survivalSelection == 'fitnessproportional':
                survivors = []
                while len(survivors) < popSize:
                    survivor = fitPropSelection(pop)
                    survivors.append(survivor)
                    pop.remove(survivor)
                pop = survivors
            elif survivalSelection == 'truncation':
                pop.sort(key = lambda x: x.fitness, reverse=True)
                pop = pop[:popSize]
            else:
                print('WARNING: survival selection method {0} not recognized. Defaulting to truncation')
                pop.sort(key=lambda x: x.fitness, reverse=True)
                pop = pop[:popSize]

        # write final results
        evalCounts = sorted(list(averageFitnessDict.keys()))
        resultWriter.writerow([str(r)])
        resultWriter.writerow(list(str(e) for e in evalCounts))
        resultWriter.writerow(list(str(averageFitnessDict[e]) for e in evalCounts))
        resultWriter.writerow(list(str(bestFitnessDict[e]) for e in evalCounts))
        resultWriter.writerow(['\n'])

        # record average and best fitnesses
        allRunsAverageFitnesses.append(averageFitnessDict)
        allRunsBestFitnesses.append(bestFitnessDict)


    # write average average, average best, best average, and best best fitnesses
    evalCounts = sorted(list(allRunsAverageFitnesses[0].keys()))

    averageAverageFitness = list((mean(allRunsAverageFitnesses[r][e] for r in range(numRuns))) for e in evalCounts)
    averageBestFitness = list((mean(allRunsBestFitnesses[r][e] for r in range(numRuns))) for e in evalCounts)
    bestAverageFitness = list((max(allRunsAverageFitnesses[r][e] for r in range(numRuns))) for e in evalCounts)
    bestBestFitness = list((max(allRunsBestFitnesses[r][e] for r in range(numRuns))) for e in evalCounts)

    resultWriter.writerow(["Avg. avg fitness, avg. best fitness, best avg. fitness, best best fitness"])
    resultWriter.writerow(evalCounts)
    resultWriter.writerow(averageAverageFitness)
    resultWriter.writerow(averageBestFitness)
    resultWriter.writerow(bestAverageFitness)
    resultWriter.writerow(bestBestFitness)

    resultFile.close()

def generateDefaultConfig(filePath):
    # generates a default configuration file and writes it to filePath
    config = configparser.ConfigParser()
    config['experiment'] = {
        'dimensionality': 30,
        'fitness function': 'rosenbrock_moderate_uniform_noise'
    }
    config['EA'] = {
        'initialization range': 10,
        'mu': 100,
        'lambda': 100,
        'maximum fitness evaluations': 3000,
        'mutation rate': 0.05,

        'parent selection algorithm': 'ktournament',
        'survival selection algorithm': 'truncation',

        'parent tournament K value': 10,
        'survival tournament K value': 10,

        'runs': 30
    }

    with open(filePath, 'w') as file:
        config.write(file)

if __name__ == "__main__":

    # set up the configuration object. This will be referenced by multiple functions and classes within this module
    config = configparser.ConfigParser()

    # read sys.argv[1] as the config file path
    # if we do not have a config file, generate and use a default config
    if len(sys.argv) < 2:
        print('No config file path provided. Generating and using default config.')
        configPath = 'default.cfg'
        generateDefaultConfig(configPath)
    # if the provided file path does not exist, generate and use a default config
    elif not os.path.isfile(sys.argv[1]):
        print('No config file found at {0}. Generating and using default config.'.format(sys.argv[1]))
        configPath = 'default.cfg'
        generateDefaultConfig(configPath)
    else:
        configPath = sys.argv[1]
        print('Using config file at {0}'.format(configPath))

    config.read(configPath)

    # record start time
    startTime = time.time()

    # seed RNGs
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    # build experiment name
    presentTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experimentName = "basicEA_" + str(presentTime)
    resultsPath = "./results/" + experimentName + '/'

    os.makedirs(resultsPath)

    # copy the used config file to the results path
    shutil.copyfile(configPath, resultsPath + 'config.cfg')

    basicEA(resultsPath)

    # print time elapsed
    print("Time elapsed: {0}".format(time.time() - startTime))
