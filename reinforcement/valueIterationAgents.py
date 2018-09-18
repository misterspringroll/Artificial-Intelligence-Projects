# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        state = mdp.getStates()
      	discount = self.discount
      	iterations = self.iterations
      	for i in range(iterations):
      		new_counter = util.Counter()
      		for state_temp in state:
      			if mdp.isTerminal(state_temp):
      				new_counter[state_temp] = 0
      			else:
      				action = self.getAction(state_temp)
      				new_counter[state_temp] = self.getQValue(state_temp,action)
      		self.values = new_counter



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """

        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        prob_group = mdp.getTransitionStatesAndProbs(state, action)
        discount = self.discount
        total_sum = 0
        for state_temp, prob in prob_group:
        	reward = mdp.getReward(state, action, state_temp)
        	next_state = discount*self.values[state_temp]
        	total_sum += prob *( reward + next_state)
        return total_sum
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        if mdp.isTerminal(state):
        	return None
        possible_actions = mdp.getPossibleActions(state)
        merged_list = []
        for action in possible_actions:
        	merged_list += [[self.computeQValueFromValues(state,action),action]]
        optimized_action = max(merged_list,key = lambda x: x[0])
        return optimized_action[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        state = mdp.getStates()
      	discount = self.discount
      	iterations = self.iterations
      	new_counter = self.values
      	i = 0
      	while i < iterations:
      		state_temp = state[i%len(state)]
      		if not mdp.isTerminal(state_temp):
      			action = self.getAction(state_temp)
      			new_counter[state_temp] = self.getQValue(state_temp,action)
      		i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        state = mdp.getStates()
        predecessor_list = {}
        discount = self.discount
      	iterations = self.iterations
      	theta = self.theta
        for temp_state in state:
        	predecessor_list[temp_state] = self.getpredecessor(temp_state)
        pq = util.PriorityQueue()
        for temp_state in state:
        	if not mdp.isTerminal(temp_state):
        		pq.push(temp_state,-self.find_difference(temp_state))
        for i in range(iterations):
        	if pq.isEmpty():
        		return
        	cur_state = pq.pop()
        	if not mdp.isTerminal(cur_state):
      			action = self.getAction(cur_state)
      			self.values[cur_state] = self.getQValue(cur_state,action)
        	for pre in predecessor_list[cur_state]:
        		diff_pre = self.find_difference(pre)
        		if diff_pre > theta:
        			pq.update(pre,-diff_pre)
    def find_difference(self,state):
    	mdp = self.mdp
    	max_q = max([self.getQValue(state,action) for action in mdp.getPossibleActions(state)])
        diff = abs(self.getValue(state)-max_q)
        return diff
    def getpredecessor(self,target_state):
    	return_list = []
    	mdp = self.mdp
    	possible_actions = mdp.getPossibleActions(target_state)
        state = mdp.getStates()
        for temp_state in state:
        	possible_actions = mdp.getPossibleActions(temp_state)
        	for temp_action in possible_actions:
        		prob_group = mdp.getTransitionStatesAndProbs(temp_state, temp_action)
        		for element in prob_group:
        			if ((element[0] == target_state) and (element[1] > 0)):
        				return_list += [temp_state]
        return return_list