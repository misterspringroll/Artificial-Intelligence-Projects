# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        #print action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currentFood = currentGameState.getFood().asList()
        food_distance_list = []
        ghost_distance = 0
        for i in range(len(currentFood)):
            food_distance_list += [manhattanDistance(newPos, currentFood[i])]
        food_distance = 100 - min(food_distance_list)
        if newScaredTimes[0] != 0:
            return food_distance
        else:
            for ghost in range(1, len(newGhostStates) + 1):

                Ghost_Position = successorGameState.getGhostPosition(ghost)
                '''print newPos
                print Ghost_Position
                print newPos == Ghost_Position'''
                x0, y0 = newPos
                x1, y1 = Ghost_Position
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if abs(i) == 1 and abs(j) == 1:
                            continue
                        if (x0 + i == x1) and (y0 + j == y1):
                            ghost_distance = -100
        if ghost_distance != 0:
            return ghost_distance
        if action == "STOP" and ghost_distance != -100:
            return -99
        if min(food_distance_list) == 0:
            return 200
        return food_distance + ghost_distance

        # minDisToGhost = 0
        # for state in newGhostStates:
        #     legalMoves = state.getLegalActions()
        #
        #     disToGhost = manhattanDistance(newPos, state.configuration.pos)
        #     if disToGhost == 1:
        #         minDisToGhost = 99999



        "*** YOUR CODE HERE ***"
        #return 0 - minDisToGhost

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.minMaxHelper(gameState, 0, 0, gameState.getNumAgents())

    def minMaxHelper2(self, gameState, agentIndex, depth, numAgents):
        if agentIndex >= numAgents:
            depth += 1
            agentIndex = 0
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return self.evaluationFunction(gameState)
        resultList1 = []
        resultList2 = []
        for action in gameState.getLegalActions(agentIndex):
            resultList1.append(float(self.minMaxHelper2(gameState.generateSuccessor(agentIndex,action), agentIndex + 1, depth, numAgents)))
            resultList2.append(action)
        if agentIndex == 0:
            return max(resultList1)
        else:
            return min(resultList1)

    def minMaxHelper(self, gameState, agentIndex, depth, numAgents):
        if agentIndex >= numAgents:
            depth += 1
            agentIndex = 0
        if gameState.isWin() or gameState.isLose() or self.depth < depth:
            return self.evaluationFunction(gameState)
        resultList1 = []
        resultList2 = []
        for action in gameState.getLegalActions(agentIndex):
            resultList1.append(float(self.minMaxHelper2(gameState.generateSuccessor(agentIndex,action), agentIndex + 1, depth, numAgents)))
            resultList2.append(action)

        if agentIndex == 0:
            return resultList2[resultList1.index(max(resultList1))]
        else:
            return resultList2[resultList1.index(min(resultList1))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeataHelper1(gameState, 0, -99999999, 99999999, 0, True)

    def alphaBeataHelper1(self,gameState, depth, a, b, agentIndex, Root):
        if agentIndex >= gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            v = -99999999
            actionRsult = "";
            for action in gameState.getLegalActions(agentIndex):
                if v < self.alphaBeataHelper1(gameState.generateSuccessor(agentIndex, action), depth, a , b, agentIndex+1, False):
                    actionRsult = action
                v = max(v, self.alphaBeataHelper1(gameState.generateSuccessor(agentIndex, action), depth, a , b, agentIndex+1, False))
                a = max(a,v)
                if b < a:
                    break
            if Root == True:
                return actionRsult
            return v
        else:
            v = 99999999
            for action in gameState.getLegalActions(agentIndex):
                v = min(v,self.alphaBeataHelper1(gameState.generateSuccessor(agentIndex, action), depth, a, b, agentIndex + 1, False))
                b = min(b, v)
                if b < a:
                    break
            return v


    # def alphaBetaHelper2(self, gameState, agentIndex, depth, numAgents, levelMinMax):
    #     if agentIndex >= numAgents:
    #         depth += 1
    #         agentIndex = 0
    #     if gameState.isWin() or gameState.isLose() or self.depth == depth:
    #         return self.evaluationFunction(gameState)
    #     resultList1 = []
    #     resultList2 = []
    #     for action in gameState.getLegalActions(agentIndex):
    #         if agentIndex == 0:
    #             if levelMinMax != None and len(resultList1) != 0 and levelMinMax < max(resultList1):
    #                 break
    #             if len(resultList1) == 0:
    #                 resultList1.append(float(
    #                     self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
    #                                           numAgents, levelMinMax)))
    #             else:
    #                 resultList1.append(float(self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,numAgents, min(resultList1))))
    #         else:
    #             if levelMinMax != None and len(resultList1) != 0 and levelMinMax > min(resultList1):
    #                 break
    #             if len(resultList1) == 0:
    #                 resultList1.append(float(self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, numAgents, levelMinMax)))
    #             else:
    #                 resultList1.append(float(
    #                     self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
    #                                           numAgents, max(resultList1))))
    #         resultList2.append(action)
    #     if agentIndex == 0:
    #         return max(resultList1)
    #     else:
    #         return min(resultList1)
    #
    # def alphaBetaHelper1(self, gameState, agentIndex, depth, numAgents):
    #     if agentIndex >= numAgents:
    #         depth += 1
    #         agentIndex = 0
    #     if gameState.isWin() or gameState.isLose() or self.depth < depth:
    #         if self.depth != depth:
    #             print 111111
    #         return self.evaluationFunction(gameState)
    #     resultList1 = []
    #     resultList2 = []
    #     for action in gameState.getLegalActions(agentIndex):
    #         if action == "Stop":
    #             continue
    #         if len(resultList1) != 0:
    #             resultList1.append(float(
    #                 self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, numAgents, max(resultList1))))
    #         else:
    #             resultList1.append(float(
    #                 self.alphaBetaHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
    #                                       numAgents, None)))
    #         resultList2.append(action)
    #
    #     if agentIndex == 0:
    #         return resultList2[resultList1.index(max(resultList1))]
    #     else:
    #         return resultList2[resultList1.index(min(resultList1))]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.ExpectimaxHelper1(gameState, 0, 0, gameState.getNumAgents())

    def ExpectimaxHelper2(self, gameState, agentIndex, depth, numAgents):
        if agentIndex >= numAgents:
            depth += 1
            agentIndex = 0
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return self.evaluationFunction(gameState)
        resultList1 = []
        resultList2 = []
        for action in gameState.getLegalActions(agentIndex):
            if action == "Stop":
                continue
            resultList1.append(float(
                self.ExpectimaxHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
                                   numAgents)))
            resultList2.append(action)
        if agentIndex == 0:
            return max(resultList1)
        else:
            return sum(resultList1) / float(len(resultList1))

    def ExpectimaxHelper1(self, gameState, agentIndex, depth, numAgents):
        if agentIndex >= numAgents:
            depth += 1
            agentIndex = 0
        if gameState.isWin() or gameState.isLose() or self.depth < depth:
            return self.evaluationFunction(gameState)
        resultList1 = []
        resultList2 = []
        for action in gameState.getLegalActions(agentIndex):
            if action == "Stop":
                continue
            resultList1.append(float(
                self.ExpectimaxHelper2(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth,
                                   numAgents)))
            resultList2.append(action)

        if agentIndex == 0:
            return resultList2[resultList1.index(max(resultList1))]
        else:
            return resultList2[resultList1.index(min(resultList1))]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The first check is to make sure we try to skip all of the failing states and
      go into the winning state. 
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
    	return 99999
    if currentGameState.isLose():
    	return -99999
    Pacman = currentGameState.getPacmanPosition()
    score = scoreEvaluationFunction(currentGameState)
    newfood = currentGameState.getFood().asList()
    GhostState = currentGameState.getGhostStates()
    food_num = len(newfood)
    closest_food = min([manhattanDistance(Pacman,food_temp) for food_temp in newfood])
    newghost = []
    for i in range(currentGameState.getNumAgents() - 1):
    	newghost += [currentGameState.getGhostPosition(i+1)]
    closest_ghost = min([manhattanDistance(Pacman,ghost_temp) for ghost_temp in newghost])
    ghost_adjust = max(closest_ghost,2)
    Active_Ghost = 0
    newcapsule = currentGameState.getCapsules()
    num_capsule = len(newcapsule)
    capsule_list = [manhattanDistance(Pacman,cap) for cap in newcapsule]
    if len(capsule_list) != 0:
    	shortest_capsule = min(capsule_list)
    else:
    	shortest_capsule = 0
    Ghost_State = currentGameState.getGhostStates()
    for ghost_temp in Ghost_State:
    	if ghost_temp.scaredTimer == 0:
    		Active_Ghost += 1
    num_ghost = len(Ghost_State)
    if closest_ghost <= 2:
    	score = score + num_capsule * 10 + 20 * shortest_capsule + 2 * ghost_adjust - 1.65 * closest_food - 4 * food_num + 2.7*(num_ghost - Active_Ghost)
    else: 
    	score = score - num_capsule * 100 + 2 * ghost_adjust - 1.65 * closest_food - 4 * food_num + 2.5*(num_ghost - Active_Ghost)
    return score

# Abbreviation
better = betterEvaluationFunction

