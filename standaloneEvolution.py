# this file supports a standalone evolution run, loading parent and survival selection functions pickled as ps and ss
# respectively, in the configured directory.
# it uses the argparse module to input configuraitons instead of a config file, and it does not support
# multiprocessing
# TODO: add this functionality to main python file instead of using standalone duplicate code

import argparse

import time
import copy
import random
import os
import csv
import math
import pickle

import numpy as np

from statistics import mean

import fitness_functions as ff


class popi:
    def __init__(self):
        self.genotype = None
        self.dataType = None

        self.fitness = None
        self.fitnessProportion = None
        self.fitnessRank = None  # higher rank = higher fitness
        self.biodiversity = None
        self.parentsFitness = None

        self.parentChance = None
        self.survivalChance = None

    def recombine(self, parent2):
        newChild = popi()
        geneLength = len(self.genotype)
        newChild.genotype = np.array(self.genotype)
        for i in range(geneLength):
            if random.random() > 0.5:
                newChild.genotype[i] = parent2.genotype[i]
        newChild.parentsFitness = (self.fitness, parent2.fitness)
        newChild.dataType = self.dataType
        return newChild

    def mutation(self):
        geneLength = len(self.genotype)
        mutationChance = args.base_ea_mutation_rate
        for i in range(geneLength):
            if random.random() < mutationChance:
                if self.dataType == 'float':
                    self.genotype[i] += random.triangular(-1, 1, 0)
                elif self.dataType == 'int':
                    self.genotype[i] += int(random.triangular(-10, 10, 0))
                elif self.dataType == 'bool':
                    self.genotype[i] = not self.genotype[i]
                else:
                    print('ERROR: data type {0} not recognized for mutation'.format(self.dataType))

    def randomize(self, initialRange, dim):
        fitness_function = args.fitness_function
        if fitness_function in ['rosenbrock_moderate_uniform_noise',
                                'rastrigin_moderate_uniform_noise']:
            self.dataType = 'float'
        elif fitness_function in ['trap',
                                  'deceptive_trap',
                                  'hierarchical_if_and_only_if']:
            self.dataType = 'bool'

        else:
            print('ERROR: no datatype found for fitness function {0}'.format(fitness_function))

        if self.dataType == 'float':
            self.genotype = (np.random.rand(dim) - 0.5) * initialRange
        elif self.dataType == 'int':
            self.genotype = list(int(p) for p in ((np.random.rand(dim) - 0.5) * initialRange))
        elif self.dataType == 'bool':
            self.genotype = list(random.random() > 0.5 for p in range(dim))
        else:
            print('ERROR: data type {0} not recognized for random initialization'.format(self.dataType))

        self.fitness = None
        self.fitnessProportion = None
        self.fitnessRank = None  # higher rank = higher fitness

        self.parentChance = None
        self.survivalChance = None


class subPopulation:
    # wrapper for a group of popis
    def __init__(self):
        self.population = None
        self.averageFitness = None
        self.bestFitness = None
        self.parentSelectionFunction = None
        self.survivalSelectionFunction = None
        self.averageFitnessDict = dict()
        self.bestFitnessDict = dict()
        self.evals = None

    def randomizeNum(self, num, initialRange, dim):
        # initializes and randomizes a number of population individuals
        self.population = []
        for i in range(num):
            newPopi = popi()
            newPopi.randomize(initialRange, dim)
            self.population.append(newPopi)

    def evaluateAll(self, populationToEval):
        function_name = args.fitness_function
        genotypes = list(p.genotype for p in populationToEval)

        fitnessValues = list(ff.get_fitness(g, function_name) for g in genotypes)

        for i, p in enumerate(populationToEval):
            p.fitness = fitnessValues[i]

    def updateStats(self):
        # Updates the fitnessRank and fitnessProportion of the population members
        self.population.sort(key=lambda p: p.fitness)

        for i, p in enumerate(self.population):
            p.fitnessRank = i

        genotypes = np.array(list(p.genotype for p in self.population))

        minFitness = self.population[0].fitness
        if minFitness < 0:
            fitAdd = 0
        else:
            fitAdd = minFitness

        fitnessSum = np.sum((p.fitness + fitAdd) for p in self.population)
        if fitnessSum == 0:
            for p in self.population:
                p.fitnessProportion = 1
        else:
            for p in self.population:
                p.fitnessProportion = (p.fitness + fitAdd) / fitnessSum

        # biodiversity
        averageGenotype = np.mean(genotypes, 0)
        genotypeDifferences = genotypes - averageGenotype
        genotypeDistances = np.sum(genotypeDifferences ** 2, 1)
        sumDistance = np.sum(genotypeDistances)
        if sumDistance != 0:
            genotypeDistances /= sumDistance
        for i, p in enumerate(self.population):
            p.biodiversity = genotypeDistances[i]

    def assignParentSelectionChances(self):
        self.updateStats()
        for p in self.population:
            terminalValues = {'fitness': p.fitness,
                              'fitnessProportion': p.fitnessProportion,
                              'fitnessRank': p.fitnessRank,
                              'biodiversity': p.biodiversity,
                              'populationSize': len(self.population),
                              'evaluations': self.evals}

            if p.parentsFitness is not None:
                terminalValues['parent1fitness'] = p.parentsFitness[0]
                terminalValues['parent2fitness'] = p.parentsFitness[1]
            else:
                terminalValues['parent1fitness'] = p.fitness
                terminalValues['parent2fitness'] = p.fitness
            p.parentChance = self.parentSelectionFunction.get(terminalValues)

        # normalize if negative chances are present
        minChance = min(p.parentChance for p in self.population)
        if minChance < 0:
            for p in self.population:
                p.parentChance -= minChance

    def assignSurvivalSelectionChances(self):
        self.updateStats()
        for p in self.population:
            terminalValues = {'fitness': p.fitness,
                              'fitnessProportion': p.fitnessProportion,
                              'fitnessRank': p.fitnessRank,
                              'biodiversity': p.biodiversity,
                              'populationSize': len(self.population),
                              'evaluations': self.evals}

            if p.parentsFitness is not None:
                terminalValues['parent1fitness'] = p.parentsFitness[0]
                terminalValues['parent2fitness'] = p.parentsFitness[1]
            else:
                terminalValues['parent1fitness'] = p.fitness
                terminalValues['parent2fitness'] = p.fitness
            p.survivalChance = self.survivalSelectionFunction.get(terminalValues)

        # normalize if negative chances are present
        minChance = min(p.survivalChance for p in self.population)
        if minChance < 0:
            for p in self.population:
                p.survivalChance -= minChance

    def parentSelection(self):
        selected = None
        totalChance = np.sum(p.parentChance for p in self.population)
        selectionNum = random.uniform(0, totalChance)
        for p in self.population:
            if selectionNum <= p.parentChance:
                selected = p
                break
            else:
                selectionNum -= p.parentChance

        if selected is None:
            print("ERROR: Overrun parent selection with " + str(selectionNum) + " remaining")
        return selected

    def survivalSelection(self):
        selected = None
        selectedIndex = None
        totalChance = np.sum(p.survivalChance for p in self.population)
        selectionNum = random.uniform(0, totalChance)
        for i, p in enumerate(self.population):
            if selectionNum <= p.survivalChance:
                selected = p
                selectedIndex = i
                break
            else:
                selectionNum -= p.survivalChance

        if selected is None:
            print("ERROR: Overrun survival selection with " + str(selectionNum) + " remaining")
        return selected, selectedIndex

    def recombine(self, parent2, mutationRate, recombinationDecider):
        # recombine/mutate the parent and survival selection functions of two subpopulations and return a new child
        newChild = subPopulation()
        newChild.parentSelectionFunction = self.parentSelectionFunction.recombine(parent2.parentSelectionFunction,
                                                                                  recombinationDecider)
        if random.random() < mutationRate:
            newChild.parentSelectionFunction.mutate()
        newChild.survivalSelectionFunction = self.survivalSelectionFunction.recombine(parent2.survivalSelectionFunction,
                                                                                      recombinationDecider)
        if random.random() < mutationRate:
            newChild.survivalSelectionFunction.mutate()
        return newChild

    def runEvolution(self):
        # runs a full EA, using the assigned parent and survival selection functions
        # the evolution parameters are read from the configuration file

        # read the parameters from the config file
        mu = args.base_ea_mu
        lam = args.base_ea_lambda

        maxEvals = args.base_ea_max_fitness_evals

        initialRange = args.init_range
        dim = args.dim

        convergenceTermination = args.convergence_termination
        convergenceGens = args.convergence_generations

        # initialization
        self.randomizeNum(mu, initialRange, dim)
        self.evaluateAll(self.population)
        self.evals = mu
        self.averageFitnessDict = dict()
        self.bestFitnessDict = dict()

        self.bestFitness = max(p.fitness for p in self.population)
        self.averageFitness = np.mean(list(p.fitness for p in self.population))

        self.averageFitnessDict[self.evals] = self.averageFitness
        self.bestFitnessDict[self.evals] = self.bestFitness

        function_name = args.fitness_function

        # EA loop
        while self.evals < maxEvals:
            # parent selection/recombination
            children = []
            self.assignParentSelectionChances()
            while len(children) < lam:
                parent1 = self.parentSelection()
                parent2 = self.parentSelection()
                newChild = parent1.recombine(parent2)
                newChild.mutation()
                children.append(newChild)
            self.evaluateAll(children)
            self.evals += lam

            # population merging
            self.population += children

            # survival selection
            survivors = []
            self.assignSurvivalSelectionChances()
            while len(survivors) < mu:
                nextSurvivor, nextSurvivorIndex = self.survivalSelection()
                survivors.append(nextSurvivor)
                self.population.pop(nextSurvivorIndex)
            self.population = survivors

            # calculate status
            self.bestFitness = max(p.fitness for p in self.population)
            self.averageFitness = np.mean(list(p.fitness for p in self.population))
            self.averageFitnessDict[self.evals] = self.averageFitness
            self.bestFitnessDict[self.evals] = self.bestFitness

            # if we have found the global optimum, terminate now
            if self.bestFitness == ff.get_optimum(self.population[0].genotype, function_name):
                # terminate early, autofilling average and best fitness dictionaries
                while self.evals < maxEvals:
                    self.evals += lam
                    self.averageFitnessDict[self.evals] = self.averageFitness
                    self.bestFitnessDict[self.evals] = self.bestFitness

            # check for early convergence termination
            if convergenceTermination and len(self.bestFitnessDict) >= convergenceGens:
                bestEvalsWindow = sorted(self.bestFitnessDict.keys())[-convergenceGens:]
                bestEvals = list(self.bestFitnessDict[e] for e in bestEvalsWindow)
                if all(b == bestEvals[0] for b in bestEvals):
                    # terminate early, autofilling average and best fitness dictionaries
                    while self.evals < maxEvals:
                        self.evals += lam
                        self.averageFitnessDict[self.evals] = self.averageFitness
                        self.bestFitnessDict[self.evals] = self.bestFitness


class GPNode:
    numericTerminals = ['constant', 'random']
    dataTerminals = ['fitness', 'fitnessProportion', 'fitnessRank', 'biodiversity', 'populationSize', 'parent1fitness',
                     'parent2fitness', 'evaluations']
    nonTerminals = ['+', '-', '*', '/', 'combo', 'step']
    childCount = {'+': 2, '-': 2, '*': 2, '/': 2, 'combo': 2, 'step': 2}

    def __init__(self):
        self.operation = None
        self.data = None
        self.children = None
        self.parent = None

    def limitedFac(self, nInput, limit=50):
        # a limited factorial function, whose max return value is limit!
        n = int(min(abs(nInput), limit))
        return math.factorial(n)

    def combo(self, n, k):
        # computes n-choose-k combination
        if n - k < 0:
            return 0
        else:
            return self.limitedFac(n) / (self.limitedFac(k) * self.limitedFac(n - k))

    def grow(self, depthLimit, parent):
        if depthLimit == 0:
            self.operation = random.choice(GPNode.numericTerminals + GPNode.dataTerminals)
        else:
            self.operation = random.choice(GPNode.numericTerminals + GPNode.dataTerminals + GPNode.nonTerminals)

        if self.operation == 'constant':
            self.data = random.expovariate(0.07)
        if self.operation in GPNode.nonTerminals:
            self.children = []
            for i in range(GPNode.childCount[self.operation]):
                newChildNode = GPNode()
                newChildNode.grow(depthLimit - 1, self)
                self.children.append(newChildNode)
        self.parent = parent

    def get(self, terminalValues):
        if self.operation == '+':
            return self.children[0].get(terminalValues) + self.children[1].get(terminalValues)
        elif self.operation == '-':
            return self.children[0].get(terminalValues) - self.children[1].get(terminalValues)
        elif self.operation == '*':
            return self.children[0].get(terminalValues) * self.children[1].get(terminalValues)
        elif self.operation == '/':
            denom = self.children[1].get(terminalValues)
            if denom == 0:
                denom = 0.00001
            return self.children[0].get(terminalValues) / denom

        elif self.operation == 'combo':
            return self.combo(self.children[0].get(terminalValues), self.children[1].get(terminalValues))

        elif self.operation == 'step':
            if self.children[0].get(terminalValues) >= self.children[1].get(terminalValues):
                return 1
            else:
                return 0

        elif self.operation in GPNode.dataTerminals:
            return terminalValues[self.operation]

        elif self.operation == 'constant':
            return self.data

        elif self.operation == 'random':
            return random.expovariate(0.07)

        else:
            print("ERROR: operation " + str(self.operation) + " not found")

    # TODO: replace this with better tree printer
    def getString(self):
        if self.operation in GPNode.nonTerminals:
            if len(self.children) == 2:
                result = "(" + self.children[0].getString() + " " + self.operation + " " + self.children[
                    1].getString() + ")"
            else:
                print('WARNING: nonterminal GP node with {0} children'.format(len(self.children)))
                result = ''
        elif self.operation == 'constant':
            result = str(self.data)
        else:
            result = self.operation
        return result

    def getAllNodes(self):
        nodes = []
        nodes.append(self)
        if self.children is not None:
            for c in self.children:
                nodes.extend(c.getAllNodes())
        return nodes

    def getAllNodesDepthLimited(self, depthLimit):
        # returns all nodes down to a certain depth limit
        nodes = []
        nodes.append(self)
        if self.children is not None and depthLimit > 0:
            for c in self.children:
                nodes.extend(c.getAllNodesDepthLimited(depthLimit - 1))
        return nodes

    def getDict(self):
        # return a dictionary containing the operation, data, and children of the node
        result = dict()
        result['data'] = self.data
        result['operation'] = self.operation
        if self.children is not None:
            result['children'] = []
            for c in self.children:
                result['children'].append(c.getDict())

    def buildFromDict(self, d):
        # builds a GPTree from a dictionary output by getDict
        self.data = d['data']
        self.operation = d['operation']
        if 'children' in d.keys():
            self.children = []
            for c in d['children']:
                newNode = GPNode()
                newNode.buildFromDict(c)
                newNode.parent = self
                self.children.append(newNode)
        else:
            self.children = None


class GPTree:
    # mostly encapsulates a tree made of GPNodes
    def __init__(self):
        self.root = None
        self.fitness = None

    def recombine(self, parent2, recombinationDecider):
        # recombines two GPTrees and returns a new child

        # copy the first parent
        newChild = copy.deepcopy(self)
        newChild.fitness = None

        # use a recombination decider if one is present, pick randomly otherwise
        if recombinationDecider is not None:
            # select a point to insert a tree from the second parent
            insertionPoint = recombinationDecider.selectRecombinationNode(newChild)

            # copy a random tree from the second parent
            replacementTree = copy.deepcopy(recombinationDecider.selectRecombinationNode(parent2))

        else:
            # select a point to insert a tree from the second parent
            insertionPoint = random.choice(newChild.getAllNodes())

            # copy a tree from the second parent
            replacementTree = copy.deepcopy(random.choice(parent2.getAllNodes()))

        # insert the tree
        newChild.replaceNode(insertionPoint, replacementTree)

        return newChild

    def mutate(self):
        # replaces a randomly selected subtree with a new random subtree

        # select a point to insert a new random tree
        insertionPoint = random.choice(self.getAllNodes())

        # randomly generate a new subtree
        newSubtree = GPNode()
        newSubtree.grow(3, None)

        # insert the new subtree
        self.replaceNode(insertionPoint, newSubtree)

    def get(self, terminalValues):
        return self.root.get(terminalValues)

    def getAllNodes(self):
        result = self.root.getAllNodes()
        return result

    def getAllNodesDepthLimited(self, depthLimit):
        result = self.root.getAllNodesDepthLimited(depthLimit)
        return result

    def replaceNode(self, nodeToReplace, replacementNode):
        # replaces node in GPTree. Uses the replacementNode directly, not a copy of it
        if nodeToReplace not in self.getAllNodes():
            print("ERROR: Attempting to replace node not in own tree")
        if nodeToReplace is self.root:
            self.root = replacementNode
            self.root.parent = None
        else:
            parentOfReplacement = nodeToReplace.parent
            for i, c in enumerate(parentOfReplacement.children):
                if c is nodeToReplace:
                    parentOfReplacement.children[i] = replacementNode
                    break
            replacementNode.parent = parentOfReplacement

    def growRoot(self, initialDepthLimit):
        if self.root is None:
            self.root = GPNode()
        self.root.grow(initialDepthLimit, None)

    def verifyParents(self):
        for n in self.getAllNodes():
            if n is self.root:
                assert (n.parent is None)
            else:
                assert (n.parent is not None)
                assert (n in n.parent.children)

    def getString(self):
        return self.root.getString()

    def getDict(self):
        return self.root.getDict()

    def buildFromDict(self, d):
        self.fitness = None
        self.root = GPNode()
        self.root.buildFromDict(d)

    def saveToDict(self, filename):
        with open(filename, 'wb') as pickleFile:
            pickle.dump(self.getDict(), pickleFile)

    def loadFromDict(self, filename):
        with open(filename, 'rb') as pickleFile:
            d = pickle.load(pickleFile)
            self.buildFromDict(d)


def setupArgs():
    # sets up and reads the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')

    parser.add_argument('--dim', type=int)
    parser.add_argument('--fitness_function')
    parser.add_argument('--init_range', type=float)
    parser.add_argument('--convergence_termination', type=bool)
    parser.add_argument('--convergence_generations', type=int)

    parser.add_argument('--base_ea_mu', type=int)
    parser.add_argument('--base_ea_lambda', type=int)
    parser.add_argument('--base_ea_mutation_rate', type=float)
    parser.add_argument('--base_ea_max_fitness_evals', type=int)

    parser.add_argument('--evolutions', type=int)
    parser.add_argument('--seed')

    args = parser.parse_args()

    return args

def main():
    resultsPath = '{0}/results/'.format(args.directory)

    # create a subpopulation and load parent and survival seleciton functions
    subPop = subPopulation()
    ps = GPTree()
    ps.loadFromDict('{0}/ps'.format(args.directory))
    ss = GPTree()
    ss.loadFromDict('{0}/ss'.format(args.directory))
    subPop.parentSelectionFunction = ps
    subPop.survivalSelectionFunction = ss
    # run evolutions
    finalBestFitnesses = []
    finalAverageFitnesses = []
    for i in range(args.evolutions):
        subPop.runEvolution()

        # write results
        with open('{0}results{1}.csv'.format(resultsPath, i), 'w') as resultsFile:
            resultWriter = csv.writer(resultsFile)
            evalNums = sorted(subPop.bestFitnessDict.keys())
            resultWriter.writerow(evalNums)
            resultWriter.writerow(subPop.averageFitnessDict[e] for e in evalNums)
            resultWriter.writerow(subPop.bestFitnessDict[e] for e in evalNums)

        finalAverageFitnesses.append(subPop.averageFitness)
        finalBestFitnesses.append(subPop.bestFitness)

    with open('{0}finalFitnesses.txt'.format(resultsPath), 'w') as finalResultsFile:
        writer = csv.writer(finalResultsFile)
        writer.writerow(finalAverageFitnesses)
        writer.writerow([mean(finalAverageFitnesses)])
        writer.writerow(finalBestFitnesses)
        writer.writerow([mean(finalBestFitnesses)])

    with open('{0}out.txt'.format(resultsPath), 'w') as outFile:
        outFile.write(str(mean(finalBestFitnesses)))

if __name__ == '__main__':
    args = setupArgs()

    if args.seed != 'time':
        seed = int(args.seed)
    else:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)

    resultsPath = '{0}/results/'.format(args.directory)
    os.makedirs(resultsPath, exist_ok=True)

    main()