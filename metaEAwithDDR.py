
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

import numpy as np

from statistics import mean

import fitness_functions as ff


class popi:
    def __init__(self):
        self.genotype = None

        self.fitness = None
        self.fitnessProportion = None
        self.fitnessRank = None  # higher rank = higher fitness

        self.parentChance = None
        self.survivalChance = None

    def recombine(self, parent2):
        newChild = popi()
        geneLength = len(self.genotype)
        newChild.genotype = np.zeros(geneLength)
        for i in range(geneLength):
            if random.random() > 0.5:
                newChild.genotype[i] = self.genotype[i]
            else:
                newChild.genotype[i] = parent2.genotype[i]
        return newChild

    def mutate(self):
        geneLength = len(self.genotype)
        for i in range(geneLength):
            self.genotype[i] += random.triangular(-1, 1, 0)
        
    def randomize(self, initialRange, dim):
        self.genotype = (np.random.rand(dim)- 0.5) * initialRange

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
        
    def randomizeNum(self, num, initialRange, dim):
        # initializes and randomizes a number of population individuals
        self.population = []
        for i in range(num):
            newPopi = popi()
            newPopi.randomize(initialRange, dim)
            self.population.append(newPopi)
            
    def evaluateAll(self, populationToEval):
        function_name = config.get('experiment', 'fitness function')
        genotypes = list(p.genotype for p in populationToEval)
        fitnessValues = list(ff.get_fitness(g, function_name) for g in genotypes)
        for i,p in enumerate(populationToEval):
            p.fitness = fitnessValues[i]

    def updateFitnessStats(self):
        # Updates the fitnessRank and fitnessProportion of the population members
        self.population.sort(key=lambda p: p.fitness)

        for i, p in enumerate(self.population):
            p.fitnessRank = i

        minFitness = min(p.fitness for p in self.population)
        if minFitness < 0:
            fitnessSum = sum((p.fitness + minFitness) for p in self.population)
            for p in self.population:
                p.fitnessProportion = (p.fitness + minFitness) / fitnessSum
        else:
            fitnessSum = sum(p.fitness for p in self.population)
            for p in self.population:
                p.fitnessProportion = p.fitness / fitnessSum

    def assignParentSelectionChances(self):
        self.updateFitnessStats()
        for p in self.population:
            terminalValues = {'fitness': p.fitness, 'fitnessProportion': p.fitnessProportion, 'fitnessRank': p.fitnessRank, 'populationSize': len(self.population)}
            p.parentChance = self.parentSelectionFunction.get(terminalValues)

        # normalize if negative chances are present
        minChance = min(p.parentChance for p in self.population)
        if minChance < 0:
            for p in self.population:
                p.parentChance -= minChance

    def assignSurvivalSelectionChances(self):
        self.updateFitnessStats()
        for p in self.population:
            terminalValues = {'fitness': p.fitness, 'fitnessProportion': p.fitnessProportion, 'fitnessRank': p.fitnessRank, 'populationSize': len(self.population)}
            p.survivalChance = self.survivalSelectionFunction.get(terminalValues)

        # normalize if negative chances are present
        minChance = min(p.survivalChance for p in self.population)
        if minChance < 0:
            for p in self.population:
                p.survivalChance -= minChance

    def parentSelection(self):
        selected = None
        totalChance = sum(p.parentChance for p in self.population)
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
        totalChance = sum(p.survivalChance for p in self.population)
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

    def recombine(self, parent2, recombinationDecider):
        # recombine the parent and survival selection functions of two subpopulations and return a new child
        newChild = subPopulation()
        newChild.parentSelectionFunction = self.parentSelectionFunction.recombine(parent2.parentSelectionFunction, recombinationDecider)
        newChild.survivalSelectionFunction = self.survivalSelectionFunction.recombine(parent2.survivalSelectionFunction, recombinationDecider)
        return newChild

    def runEvolution(self):
        # runs a full EA, using the assigned parent and survival selection functions
        # the evolution parameters are read from the configuration file

        # read the parameters from the config file
        mu = config.getint('baseEA', 'base EA mu')
        lam = config.getint('baseEA', 'base EA lambda')
        maxEvals = config.getint('baseEA', 'base EA maximum fitness evaluations')
        initialRange = config.getint('baseEA', 'initialization range')
        mutationRate = config.getfloat('baseEA', 'base EA mutation rate')
        dim = config.getint('experiment', 'dimensionality')

        convergenceTermination = config.getboolean('baseEA', 'convergence termination')
        convergenceGens = config.getint('baseEA', 'convergence generations')

        # initialization
        self.randomizeNum(mu, initialRange, dim)
        self.evaluateAll(self.population)
        evals = mu
        self.averageFitnessDict = dict()
        self.bestFitnessDict = dict()

        self.bestFitness = max(p.fitness for p in self.population)
        self.averageFitness = sum(p.fitness for p in self.population) / len(self.population)

        self.averageFitnessDict[evals] = self.averageFitness
        self.bestFitnessDict[evals] = self.bestFitness

        # EA loop
        while evals < maxEvals:
            # parent selection/recombination
            children = []
            self.assignParentSelectionChances()
            while len(children) < lam:
                parent1 = self.parentSelection()
                parent2 = self.parentSelection()
                newChild = parent1.recombine(parent2)
                if random.random() < mutationRate:
                    newChild.mutate()
                children.append(newChild)
            self.evaluateAll(children)
            evals += lam

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
            self.averageFitness = sum(p.fitness for p in self.population) / len(self.population)
            self.averageFitnessDict[evals] = self.averageFitness
            self.bestFitnessDict[evals] = self.bestFitness

            # check for early termination
            if convergenceTermination and len(self.bestFitnessDict) >= convergenceGens:
                bestEvalsWindow = sorted(self.bestFitnessDict.keys())[-convergenceGens:]
                bestEvals = list(self.bestFitnessDict[e] for e in bestEvalsWindow)
                if all(b == bestEvals[0] for b in bestEvals):
                    # terminate early, autofilling average and best fitness dictionaries
                    while evals < maxEvals:
                        self.averageFitnessDict[evals] = self.averageFitness
                        self.bestFitnessDict[evals] = self.bestFitness
                        evals += lam

class GPNode:
    numericTerminals = ['constant']
    dataTerminals = ['fitness', 'fitnessProportion', 'fitnessRank', 'populationSize']
    nonTerminals = ['+', '-', '*', '/', 'combo']
    childCount = {'+': 2, '-': 2, '*': 2, '/': 2, 'combo': 2}

    def __init__(self):
        self.operation = None
        self.data = None
        self.children = None
        self.parent = None

    def limitedFac(self, nInput, limit=50):
        # a limited factorial function, whose max return value is limit!
        n = min(abs(int(nInput)), limit)
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
            self.data = random.randrange(0, 10)
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

        elif self.operation in GPNode.dataTerminals:
            return terminalValues[self.operation]

        elif self.operation == 'constant':
            return self.data

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

class GPRecombinationDecider:
    # an object that manages recombination of GP trees to facilitate dynamic decomposition and recomposition of GP primitives
    def __init__(self):
        # the size of the sliding window used to determine whether fitness is stagnating
        self.fitnessWindowSize = config.getint('DDR', 'fitness window size')
        # the maximum amount of fitness change throughout the fitness window that will trigger a recombination depth increase
        self.fitnessStagnationThreshold = config.getfloat('DDR', 'fitness stagnation threshold')
        # the minimum amount of fitness change anywhere in the fitness window that will trigger a recombination depth decrease
        self.fitnessClimbThreshold = config.getfloat('DDR', 'fitness climb threshold')

        # the distance from the root node within which recombination is permitted
        self.recombinationDepth = config.getint('DDR', 'initial recombination depth')

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

def metaEAoneRun(runNum):
    # runs the meta EA for one run

    # get the parameters from the configuration file
    dim = config.getint('experiment', 'dimensionality')

    GPMu = config.getint('metaEA', 'metaEA mu')
    GPLambda = config.getint('metaEA', 'metaEA lambda')
    maxGPEvals = config.getint('metaEA', 'metaEA maximum fitness evaluations')
    initialGPDepthLimit = config.getint('metaEA', 'metaEA GP tree initialization depth limit')
    GPKTournamentK = config.getint('metaEA', 'metaEA k-tournament size')

    DDREnabled = config.getboolean('DDR', 'DDR enabled')

    numFinalEARuns = config.getint('metaEA', 'base EA runs') #TODO:make this more configurable

    # initialize the recombination decider if we are using one
    if DDREnabled:
        recombinationDecider = GPRecombinationDecider()
    else:
        recombinationDecider = None

    # initialize a list to keep track of recombination depths
    recombinationDepth = []

    # initialize the subpopulations
    GPPopulation = []
    for i in range(GPMu):
        newSubPop = subPopulation()
        newSubPop.parentSelectionFunction = GPTree()
        newSubPop.parentSelectionFunction.growRoot(initialGPDepthLimit)
        newSubPop.survivalSelectionFunction = GPTree()
        newSubPop.survivalSelectionFunction.growRoot(initialGPDepthLimit)
        newSubPop.runEvolution()
        GPPopulation.append(newSubPop)

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
        # parent selection (k tournament)
        children = []
        while len(children) < GPLambda:
            parent1 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.bestFitness)
            parent2 = max(random.sample(GPPopulation, GPKTournamentK), key=lambda p: p.bestFitness)
            newChild = parent1.recombine(parent2, recombinationDecider)
            newChild.runEvolution()
            GPEvals += 1
            children.append(newChild)

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
        finalEA.runEvolution()
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

    resultFile.close()

    return averageAverageFitness, averageBestFitness, bestAverageFitness, bestBestFitness, recombinationDepth

def metaEAWithDDR(resultsPath):
    numMetaRuns = config.getint('experiment', 'metaEA runs')

    subPopMu = config.getint('baseEA', 'base EA mu')
    subPopLambda = config.getint('baseEA', 'base EA lambda')

    maxSubPopEvals = config.getint('baseEA', 'base EA maximum fitness evaluations')

    DDREnabled = config.getboolean('DDR', 'DDR enabled')

    GPMu = config.getint('metaEA', 'metaEA mu')
    GPLambda = config.getint('metaEA', 'metaEA lambda')
    maxGPEvals = config.getint('metaEA', 'metaEA maximum fitness evaluations')

    usingMultiprocessing = True

    if usingMultiprocessing:

        # create the process pool to run the metaEA runs
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

    evalCounts = range(subPopMu, maxSubPopEvals + 1, subPopLambda)
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
        'GP initialization depth limit': 3
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
        'base EA runs': 30
    }
    config['baseEA'] = {
        'initialization range': 10,
        'base EA mu': 100,
        'base EA lambda': 100,
        'base EA maximum fitness evaluations': 3000,
        'base EA mutation rate': 0.05,
        'convergence termination': True,
        'convergence generations': 5
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

    DDREnabled = config.getboolean('DDR', 'DDR enabled')

    dim = config.getint('experiment', 'dimensionality')

    # record start time
    startTime = time.time()

    # seed RNGs
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
    print("Time elapsed: {0}".format(time.time() - startTime))

