
import sys
import time
import copy
import random
import datetime
import os
import csv
import math
import multiprocessing
import configparser
import shutil
import functools
import pickle
import subprocess

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
        mutationChance = getFromConfig('baseEA', 'base EA mutation rate', 'float')
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
        fitness_function = getFromConfig('experiment', 'fitness function')
        if fitness_function in ['rosenbrock_moderate_uniform_noise',
                                'rastrigin_moderate_uniform_noise']:
            self.dataType = 'float'
        elif fitness_function in ['trap',
                                  'deceptive_trap',
                                  'hierarchical_if_and_only_if']:
            self.dataType = 'bool'

        else:
            print('ERROR: no datatype found for fitness function {0}'.format(config['experiment']['fitness function']))

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
        self.processingPool = None

    def setupPool(self):
        if self.processingPool is not None:
            self.processingPool.close()
            self.processingPool = None

        numProcesses = getFromConfig('baseEA', 'base EA fitness function processes', 'int')

        if numProcesses != 1:
            if numProcesses > 1:
                self.processingPool = multiprocessing.Pool(processes=numProcesses)
            else:
                self.processingPool = multiprocessing.Pool()
        else:
            self.processingPool = None
        
    def randomizeNum(self, num, initialRange, dim):
        # initializes and randomizes a number of population individuals
        self.population = []
        for i in range(num):
            newPopi = popi()
            newPopi.randomize(initialRange, dim)
            self.population.append(newPopi)
            
    def evaluateAll(self, populationToEval):
        function_name = getFromConfig('experiment', 'fitness function')
        genotypes = list(p.genotype for p in populationToEval)

        if self.processingPool is not None:
            fitnessValues = self.processingPool.starmap(ff.get_fitness, ((g, function_name) for g in genotypes))
        else:
            fitnessValues = list(ff.get_fitness(g, function_name) for g in genotypes)

        for i,p in enumerate(populationToEval):
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
        newChild.parentSelectionFunction = self.parentSelectionFunction.recombine(parent2.parentSelectionFunction, recombinationDecider)
        if random.random() < mutationRate:
            newChild.parentSelectionFunction.mutate()
        newChild.survivalSelectionFunction = self.survivalSelectionFunction.recombine(parent2.survivalSelectionFunction, recombinationDecider)
        if random.random() < mutationRate:
            newChild.survivalSelectionFunction.mutate()
        return newChild

    def runEvolution(self, final=False):
        # runs a full EA, using the assigned parent and survival selection functions
        # the evolution parameters are read from the configuration file

        # read the parameters from the config file
        mu = getFromConfig('baseEA', 'base EA mu', 'int')
        lam = getFromConfig('baseEA', 'base EA lambda', 'int')

        if not final:
            maxEvals = getFromConfig('baseEA', 'base EA maximum fitness evaluations', 'int')
        else:
            maxEvals = getFromConfig('baseEA', 'base EA final run maximum fitness evaluations', 'int')

        initialRange = getFromConfig('baseEA', 'initialization range', 'int')
        dim = getFromConfig('experiment', 'dimensionality', 'int')

        convergenceTermination = getFromConfig('baseEA', 'convergence termination', 'bool')
        convergenceGens = getFromConfig('baseEA', 'convergence generations', 'int')

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

        self.setupPool()

        function_name = getFromConfig('experiment', 'fitness function')

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
            if convergenceTermination and len(self.bestFitnessDict) >= convergenceGens and not final:
                bestEvalsWindow = sorted(self.bestFitnessDict.keys())[-convergenceGens:]
                bestEvals = list(self.bestFitnessDict[e] for e in bestEvalsWindow)
                if all(b == bestEvals[0] for b in bestEvals):
                    # terminate early, autofilling average and best fitness dictionaries
                    while self.evals < maxEvals:
                        self.evals += lam
                        self.averageFitnessDict[self.evals] = self.averageFitness
                        self.bestFitnessDict[self.evals] = self.bestFitness

        # close fitness evaluation pool
        if self.processingPool is not None:
            self.processingPool.close()
            self.processingPool = None

class GPNode:
    numericTerminals = ['constant', 'random']
    dataTerminals = ['fitness', 'fitnessProportion', 'fitnessRank', 'biodiversity', 'populationSize', 'parent1fitness', 'parent2fitness', 'evaluations']
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
                newChildNode.grow(depthLimit-1, self)
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
                result = "(" + self.children[0].getString() + " " + self.operation + " " + self.children[1].getString() + ")"
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
        return result

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
                assert(n.parent is None)
            else:
                assert(n.parent is not None)
                assert(n in n.parent.children)

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

class GPRecombinationDecider:
    # an object that manages recombination of GP trees to facilitate dynamic decomposition and recomposition of GP primitives
    def __init__(self):
        # the size of the sliding window used to determine whether fitness is stagnating
        self.fitnessWindowSize = getFromConfig('DDR', 'fitness window size', 'int')
        # the maximum amount of fitness change throughout the fitness window that will trigger a recombination depth increase
        self.fitnessStagnationThreshold = getFromConfig('DDR', 'fitness stagnation threshold', 'float')
        # the minimum amount of fitness change anywhere in the fitness window that will trigger a recombination depth decrease
        self.fitnessClimbThreshold = getFromConfig('DDR', 'fitness climb threshold', 'float')

        # the distance from the root node within which recombination is permitted
        self.recombinationDepth = getFromConfig('DDR', 'initial recombination depth', 'int')

        # the number of generations since the last change in recombinationDepth
        self.gensSinceLastDepthChange = 0
        # a history of best fitness values for the evolution
        self.bestFitnessHistory = []
        # history of average fitness values
        self.averageFitnessHistory = []

    def adjustRecombinationDepth(self, population):
        # adjusts the current recombination depth based on fitness history and current population fitness
        # update the fitness records with the current best fitnesses of each subpopulation
        currentAverageFitness = sum(p.bestFitness for p in population) / len(population)
        currentBestFitness = max(p.bestFitness for p in population)
        self.averageFitnessHistory.append(currentAverageFitness)
        self.bestFitnessHistory.append(currentBestFitness)

        depthChangedThisTime = False

        # don't try to adjust if we don't have enough fitness information since the last reset
        if self.gensSinceLastDepthChange >= self.fitnessWindowSize:
            # check if our best fitness is stagnating
            if all(abs(self.bestFitnessHistory[-i] - self.bestFitnessHistory[-i - 1]) <= self.fitnessStagnationThreshold for i in range(1, self.fitnessWindowSize - 1)):
                self.recombinationDepth += 1
                depthChangedThisTime = True
            elif any(abs(self.bestFitnessHistory[-i] - self.bestFitnessHistory[-i - 1]) >= self.fitnessClimbThreshold for i in range(1, self.fitnessWindowSize - 1)):
                self.recombinationDepth -= 1
                depthChangedThisTime = True

        # reset or increment counter
        if depthChangedThisTime:
            self.gensSinceLastDepthChange = 0
        else:
            self.gensSinceLastDepthChange += 1

    def selectRecombinationNode(self, tree):
        # selects a node in the tree for recombination
        result = random.choice(tree.getAllNodesDepthLimited(self.recombinationDepth))
        return result

def setupIraceFiles(directory):
    dim = getFromConfig('experiment', 'dimensionality', 'int')
    fitnessFunction = getFromConfig('experiment', 'fitness function')
    convergenceTermination = getFromConfig('baseEA', 'convergence termination')
    convergenceGenerations = getFromConfig('baseEA', 'convergence generations')
    initRange = getFromConfig('baseEA', 'initialization range', 'int')
    fitnessEvals = getFromConfig('baseEA', 'base ea maximum fitness evaluations')

    with open(directory + '/scenario.txt', 'w') as scenarioFile:
        scenarioFile.write('maxExperiments = 300')

    with open(directory + '/parameters.txt', 'w') as parametersFile:
        parametersFile.write(
        'base_ea_mu   "--base_ea_mu " i (2, 100) \n \
        base_ea_lambda  "--base_ea_lambda " i (1, 100) \n \
        base_ea_mutation_rate "--base_ea_mutation_rate " r (0, 1) \n'
        )

    with open(directory + '/target-runner', 'w') as targetRunnerFile:
        targetRunnerFile.write('#!/bin/bash \n \
        EXE="python3 ../standaloneEvolution.py" \n')
        targetRunnerFile.write('FIXED_PARAMS="--dim {0} --directory . --fitness_function {1} --convergence_termination {2} --convergence_generations {3} --evolutions 1 --init_range {4} --base_ea_max_fitness_evals {5}" \n'.format(dim, fitnessFunction, convergenceTermination, convergenceGenerations, initRange, fitnessEvals))
        targetRunnerFile.write(
        'CONFIG_ID=$1 \n \
        INSTANCE_ID=$2 \n \
        SEED=$3 \n \
        INSTANCE=$4 \n \
        shift 4 || exit 1 \n \
        CONFIG_PARAMS=$* \n \
        \n \
        STDOUT=c${CONFIG_ID}-${INSTANCE_ID}.stdout \n \
        STDERR=c${CONFIG_ID}-${INSTANCE_ID}.stderr \n \
        \n \
        OUTPUT="out.txt" \n \
        \n \
        $EXE ${FIXED_PARAMS} --seed ${SEED} ${CONFIG_PARAMS} \n \
        \n \
        error() { \n \
            echo "`TZ=UTC date`: error: $@" \n \
            exit 1 \n \
        } \n \
        \n \
        if [ -s "${OUTPUT}" ]; then \n \
            COST=$(tail -n 1 ${OUTPUT} | grep -e "^[[:space:]]*[+-]\?[0-9]" | cut -f1) \n \
            echo "$COST" \n \
            exit 0 \n \
        else \n \
            error "${OUTPUT}: No such file or directory" \n \
        fi \n ')

    subprocess.run('chmod +x target-runner', cwd=directory, shell=True)


def runSingleStandaloneInDirectory(directory):
    iraceHome = getFromConfig('experiment', 'irace home')
    iraceCommand = iraceHome + '/bin/irace'

    output = str(subprocess.check_output(iraceCommand, cwd=directory))

    output = output.split('\n')
    tunedArgs = ''
    for i, o in enumerate(output):
        if '# Best configurations as commandlines (first number is the configuration ID; same order as above):' in o:
            tunedArgs = output[i+1]

    tunedArgs = tunedArgs.split(' ')
    tunedArgs = ' '.join(tunedArgs[1:])

    dim = getFromConfig('experiment', 'dimensionality')
    fitnessFunction = getFromConfig('experiment', 'fitness function')
    convergenceTermination = getFromConfig('baseEA', 'convergence termination')
    convergenceGenerations = getFromConfig('baseEA', 'convergence generations')
    initRange = getFromConfig('baseEA', 'initialization range')
    fitnessEvals = getFromConfig('baseEA', 'base ea maximum fitness evaluations')
    evolutions = getFromConfig('metaEA', 'base ea runs')
    seed = getFromConfig('experiment', 'seed')

    evolutionCommand = 'python3 standaloneEvolution.py ' \
                       '--directory {0} ' \
                       '--dim {1} ' \
                       '--fitness_function {2} ' \
                       '--init_range {3} ' \
                       '--convergence_termination {4} ' \
                       '--convergence_generations {5} ' \
                       '--base_ea_max_fitness_evals {6} ' \
                       '--evolutions {7} ' \
                       '--seed {8} ' \
                       ' {9} '.format(directory,
                                     dim,
                                     fitnessFunction,
                                     initRange,
                                     convergenceTermination,
                                     convergenceGenerations,
                                     fitnessEvals,
                                     evolutions,
                                     seed,
                                     tunedArgs)

    subprocess.run(evolutionCommand)

    with open(directory + '/out.txt') as out:
        readOut = out.read()
        readOut = readOut.split('\n')
        finalMeanAverageFitness = float(readOut[-2])
        finalMeanBestFitness = float(readOut[-1])

    return finalMeanAverageFitness, finalMeanBestFitness


def runStandaloneEvolution(GPPopulation):
    directories = []

    for i in range(len(GPPopulation)):
        subDirectory = './{0}'.format(i)
        if os.path.exists(subDirectory):
            shutil.rmtree(subDirectory)
        os.makedirs(subDirectory)

        setupIraceFiles(subDirectory)
        psFilename = subDirectory + '/ps'
        ssFilename = subDirectory + '/ss'

        GPPopulation[i].parentSelectionFunction.saveToDict(psFilename)
        GPPopulation[i].survivalSelectionFunction.saveToDict(ssFilename)

        # the tuner requires an instances directory with at least one instance in it
        instancesPath = subDirectory + '/Instances'
        os.makedirs(instancesPath)
        with open(instancesPath + '/blank.txt', 'w') as blank:
            blank.write('blank')

        directories.append(subDirectory)

    processes = getFromConfig('metaEA', 'tuning processes', 'int')

    if processes != 1:
        if processes == -1:
            processingPool = multiprocessing.Pool()
        else:
            processingPool = multiprocessing.Pool(processes=processes)

        results = processingPool.map(runSingleStandaloneInDirectory, directories)

    else:
        results = []
        for d in directories:
            results.append(runSingleStandaloneInDirectory(d))

    for i, r in enumerate(results):
        finalMeanAverageFitness, finalMeanBestFitness = r
        GPPopulation[i].averageFitness = finalMeanAverageFitness
        GPPopulation[i].bestFitness = finalMeanBestFitness

    for d in directories:
        shutil.rmtree(d)


def metaEAoneRun(runNum):
    # runs the meta EA for one run

    # get the parameters from the configuration file
    GPMu = getFromConfig('metaEA', 'metaEA mu', 'int')
    GPLambda = getFromConfig('metaEA', 'metaEA lambda', 'int')
    maxGPEvals = getFromConfig('metaEA', 'metaEA maximum fitness evaluations', 'int')
    initialGPDepthLimit = getFromConfig('metaEA', 'metaEA GP tree initialization depth limit', 'int')
    GPKTournamentK = getFromConfig('metaEA', 'metaEA k-tournament size', 'int')

    DDREnabled = getFromConfig('DDR', 'DDR enabled', 'bool')

    GPmutationRate = getFromConfig('metaEA', 'metaEA mutation rate', 'float')

    numFinalEARuns = getFromConfig('metaEA', 'base EA runs', 'int')

    useIraceTuning = getFromConfig('experiment', 'use irace tuning')
    iraceHome = getFromConfig('experiment', 'irace home')

    # initialize the recombination decider if we are using one
    if DDREnabled:
        recombinationDecider = GPRecombinationDecider()
    else:
        recombinationDecider = None

    # initialize a list to keep track of recombination depths
    recombinationDepth = []

    # initialize the subpopulations
    GPPopulation = []
    toBeEvaluated = []
    for i in range(GPMu):
        newSubPop = subPopulation()
        newSubPop.parentSelectionFunction = GPTree()
        newSubPop.parentSelectionFunction.growRoot(initialGPDepthLimit)
        newSubPop.survivalSelectionFunction = GPTree()
        newSubPop.survivalSelectionFunction.growRoot(initialGPDepthLimit)
        if useIraceTuning:
            toBeEvaluated.append(newSubPop)
        else:
            newSubPop.runEvolution()
        GPPopulation.append(newSubPop)

    if useIraceTuning:
        runStandaloneEvolution(toBeEvaluated)
        toBeEvaluated = []

    # update the recombination decider (if we have one), and record the new recombination depth
    if recombinationDecider is not None:
        recombinationDecider.adjustRecombinationDepth(GPPopulation)
        recombinationDepth.append(recombinationDecider.recombinationDepth)

    GPEvals = GPMu

    # print status
    progress = round(float(GPEvals) * 100 / maxGPEvals)
    print("Run " + str(runNum) + " progress: " + str(progress) + "%")

    # GP EA loop
    while GPEvals < maxGPEvals:

        children = []
        while len(children) < GPLambda:
            # parent selection (k tournament)
            parent1 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.bestFitness)
            parent2 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.bestFitness)
            # recombination/mutation
            newChild = parent1.recombine(parent2, GPmutationRate, recombinationDecider)
            # run evolution with this child
            if useIraceTuning:
                toBeEvaluated.append(newChild)
            else:
                newChild.runEvolution()
            GPEvals += 1
            # add to the population
            children.append(newChild)

        if useIraceTuning:
            runStandaloneEvolution(toBeEvaluated)
            toBeEvaluated = []

        # population merging
        GPPopulation += children

        # survival selection
        GPPopulation.sort(key=lambda p: p.bestFitness, reverse=True)
        GPPopulation = GPPopulation[:GPMu]

        # update the recombination decider
        if recombinationDecider is not None:
            recombinationDecider.adjustRecombinationDepth(GPPopulation)
            recombinationDepth.append(recombinationDecider.recombinationDepth)

        # print status
        progress = round(float(GPEvals) * 100 / maxGPEvals)
        print("Run " + str(runNum) + " progress: " + str(progress) + "%")

    # test the best evolution of the GPPopulation
    finalEA = max(GPPopulation, key=lambda p: p.bestFitness)
    resultFile = open(resultsPath + "results" + str(runNum) + ".csv", 'w')
    resultWriter = csv.writer(resultFile)

    allRunsAverageFitnesses = []
    allRunsBestFitnesses = []

    for r in range(numFinalEARuns):
        finalEA.runEvolution(final=True)
        evalCounts = sorted(list(finalEA.averageFitnessDict.keys()))

        allRunsAverageFitnesses.append(finalEA.averageFitnessDict)
        allRunsBestFitnesses.append(finalEA.bestFitnessDict)

        resultWriter.writerow([str(r)])
        resultWriter.writerow(list(str(e) for e in evalCounts))
        resultWriter.writerow(list(str(finalEA.averageFitnessDict[e]) for e in evalCounts))
        resultWriter.writerow(list(str(finalEA.bestFitnessDict[e]) for e in evalCounts))
        resultWriter.writerow(['\n'])

    evalCounts = sorted(list(allRunsAverageFitnesses[0].keys()))

    averageAverageFitness = list((mean(allRunsAverageFitnesses[r][e] for r in range(numFinalEARuns))) for e in evalCounts)
    averageBestFitness = list((mean(allRunsBestFitnesses[r][e] for r in range(numFinalEARuns))) for e in evalCounts)
    bestAverageFitness = list((max(allRunsAverageFitnesses[r][e] for r in range(numFinalEARuns))) for e in evalCounts)
    bestBestFitness = list((max(allRunsBestFitnesses[r][e] for r in range(numFinalEARuns))) for e in evalCounts)

    resultWriter.writerow(["Avg. avg fitness, avg. best fitness, best avg. fitness, best best fitness"])
    resultWriter.writerow(evalCounts)
    resultWriter.writerow(averageAverageFitness)
    resultWriter.writerow(averageBestFitness)
    resultWriter.writerow(bestAverageFitness)
    resultWriter.writerow(bestBestFitness)

    if recombinationDecider is not None:
        resultWriter.writerow(['Recombination Depth'])
        resultWriter.writerow(evalCounts)
        resultWriter.writerow(recombinationDepth)

    # write the best parent and survival selections to the file
    bestGPPopi = max(GPPopulation, key=lambda p: p.bestFitness)
    resultWriter.writerow([bestGPPopi.parentSelectionFunction.getString()])
    resultWriter.writerow([bestGPPopi.survivalSelectionFunction.getString()])

    # record the best population member
    bestSubPop = max(GPPopulation, key=lambda p:p.bestFitness)
    bestPopi = max(bestSubPop.population, key=lambda p:p.fitness)
    resultWriter.writerow(['Best population member:'])
    resultWriter.writerow(list(bestPopi.genotype))

    resultFile.close()

    return averageAverageFitness, averageBestFitness, bestAverageFitness, bestBestFitness, recombinationDepth

def metaEAWithDDR(resultsPath):
    numMetaRuns = getFromConfig('experiment', 'metaEA runs', 'int')

    subPopMu = getFromConfig('baseEA', 'base EA mu', 'int')
    subPopLambda = getFromConfig('baseEA', 'base EA lambda', 'int')

    maxSubPopEvalsFinalRun = getFromConfig('baseEA', 'base EA final run maximum fitness evaluations', 'int')

    DDREnabled = getFromConfig('DDR', 'DDR enabled', 'bool')

    GPMu = getFromConfig('metaEA', 'metaEA mu', 'int')
    GPLambda = getFromConfig('metaEA', 'metaEA lambda', 'int')
    maxGPEvals = getFromConfig('metaEA', 'metaEA maximum fitness evaluations', 'int')

    numProcesses = getFromConfig('metaEA', 'metaEA processes', 'int')

    if numProcesses > 1 or numProcesses == -1:

        # create the process pool to run the metaEA runs
        if numProcesses > 1:
            processPool = multiprocessing.Pool(processes=numProcesses)
        else:
            processPool = multiprocessing.Pool()

        # run the metaEA runs
        results = processPool.map(metaEAoneRun, range(numMetaRuns))

    else:
        results = []
        for i in range(numMetaRuns):
            results.append(metaEAoneRun(i))

    # calculate the overall results
    averageAverageAverage = []
    averageAverageBest = []
    averageBestAverage = []
    averageBestBest = []
    bestAverageAverage = []
    bestAverageBest = []
    bestBestAverage = []
    bestBestBest = []
    averageRecombinationDepth = []

    evalCounts = range(subPopMu, maxSubPopEvalsFinalRun + 1, subPopLambda)
    GPevalCounts = range(GPMu, maxGPEvals + 1, GPLambda)

    for i in range(len(evalCounts)):
        aa = list(r[0][i] for r in results)
        averageAverageAverage.append(mean(aa))
        bestAverageAverage.append(max(aa))

        ab = list(r[1][i] for r in results)
        averageAverageBest.append(mean(ab))
        bestAverageBest.append(max(ab))

        ba = list(r[2][i] for r in results)
        averageBestAverage.append(mean(ba))
        bestBestAverage.append(max(ba))

        bb = list(r[3][i] for r in results)
        averageBestBest.append(mean(bb))
        bestBestBest.append(max(bb))

    if DDREnabled:
        for i in range(len(GPevalCounts)):
            depth = list(r[4][i] for r in results)
            averageRecombinationDepth.append(mean(depth))

    with open(resultsPath + 'finalResults.csv', 'w') as finalResultsFile:
        finalResultsWriter = csv.writer(finalResultsFile)
        finalResultsWriter.writerow(['avg. avg. avg.',
                                  'avg. avg. best',
                                  'avg. best avg.',
                                  'avg. best best',
                                  'best avg. avg.',
                                  'best avg. best',
                                  'best best avg.',
                                  'best best best'])
        finalResultsWriter.writerow(evalCounts)
        finalResultsWriter.writerow(averageAverageAverage)
        finalResultsWriter.writerow(averageAverageBest)
        finalResultsWriter.writerow(averageBestAverage)
        finalResultsWriter.writerow(averageBestBest)
        finalResultsWriter.writerow(bestBestAverage)
        finalResultsWriter.writerow(bestAverageBest)
        finalResultsWriter.writerow(bestBestAverage)
        finalResultsWriter.writerow(bestBestBest)

        if DDREnabled:
            finalResultsWriter.writerow(['avg. recombination depth'])
            finalResultsWriter.writerow(GPevalCounts)
            finalResultsWriter.writerow(averageRecombinationDepth)

def generateDefaultConfig(filePath):
    # generates a default configuration file and writes it to filePath
    config = configparser.ConfigParser()
    config['experiment'] = {
        'metaEA runs': 30,
        'dimensionality': 30,
        'fitness function': 'rosenbrock_moderate_uniform_noise',
        'GP initialization depth limit': 3,
        'seed': 'time',
        'use irace tuning': True,
        'irace home': '~/R/x86_64-pc-linux-gnu-library/3.2/irace/'
    }
    config['DDR'] = {
        'DDR enabled': True,
        'fitness window size': 3,
        'fitness stagnation threshold': 100,
        'fitness climb threshold': 1000,
        'initial recombination depth': 6
    }
    config['metaEA'] = {
        'metaEA mu': 20,
        'metaEA lambda': 10,
        'metaEA maximum fitness evaluations': 200,
        'metaEA k-tournament size': 8,
        'metaEA GP tree initialization depth limit': 3,
        'metaEA mutation rate': 0.01,
        'base EA runs': 30,
        'processes': -1,
        'tuning processes': -1
    }
    config['baseEA'] = {
        'initialization range': 10,
        'base EA mu': 100,
        'base EA lambda': 100,
        'base EA maximum fitness evaluations': 3000,
        'base EA final run maximum fitness evaluations': 1000000,
        'base EA mutation rate': 0.05,
        'convergence termination': True,
        'convergence generations': 5
    }

    with open(filePath, 'w') as file:
        config.write(file)

@functools.lru_cache(maxsize=2048)
def getFromConfig(section, value, type=None):
    if type == 'bool':
        return config.getboolean(section, value)
    elif type == 'int':
        return config.getint(section, value)
    elif type == 'float':
        return config.getfloat(section, value)
    else:
        return config.get(section, value)

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

    DDREnabled = getFromConfig('DDR', 'DDR enabled', 'bool')

    dim = getFromConfig('experiment', 'dimensionality', 'int')

    # record start time
    startTime = time.time()

    # seed RNGs
    seed = getFromConfig('experiment', 'seed')
    try:
        seed = int(seed)
    except ValueError:
        seed = int(time.time())

    random.seed(seed)
    np.random.seed(seed)

    # build experiment name
    presentTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experimentName = "metaEAwithDDR_" + str(presentTime)
    resultsPath = "./results/" + experimentName + '/'

    os.makedirs(resultsPath)

    # copy the used config file to the results path
    shutil.copyfile(configPath, resultsPath + 'config.cfg')

    # run experiment
    metaEAWithDDR(resultsPath)

    # print time elapsed
    timeElapsed = time.time() - startTime
    print("Time elapsed: {0}".format(timeElapsed))

    # record seed and time elapsed
    with open(resultsPath + 'log.txt', 'w') as logFile:
        logFile.write('Random seed used: {0}\n'.format(seed))
        logFile.write('Time elapsed: {0}\n'.format(timeElapsed))
