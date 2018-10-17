#!/usr/bin/env python
'''Modules for support classes.'''

import time
import copy
import numpy as np
from collections import defaultdict

# libraries for logging
from debug_logger import setupLogging
import logging

# set up module logging
setupLogging()
logger = logging.getLogger(__name__)

class state(object):
    '''State data object to represent all states'''
    def __init__(self, x, y, heading):
        '''Constructor

        Arguments:
            x - int: x coordinate of state
            y - int: y coordinate of state
            heading - int: heading of state
        '''
        self.x = x
        self.y = y
        self.heading = heading

    def __eq__(self, another):
        '''Override equality such that any state with the same properties is equal'''
        return (hasattr(another, 'x') and self.x == another.x) \
            and (hasattr(another, 'y') and self.y == another.y) \
            and (hasattr(another, 'heading') and self.heading == another.heading)

    def __hash__(self):
        '''Override hashing such that different instances can give the same key'''
        return hash((self.x,self.y,self.heading))

    def __repr__(self):
        '''Override to allow easier printing of states'''
        return 'state' + str((self.x, self.y, self.heading))

class action(object):
    '''Action data object to represent all actions'''
    def __init__(self, movement, rotation=None, vector=None):
        '''Constructor
        
        Arguments:
            movement - string: describes movement of robot ['forwards','backwards','stay']
            rotation - string: describes rotation of robot ['right', 'left', None]
            vector - 1D numpy array: direction vector of movement with action('forwards',None)
                defined as positive y direction or as np.array([[0],[1]])
        '''
        self.movement = movement
        self.rotation = rotation
        self.vector = vector

    def __eq__(self, another):
        '''Override equality such that any action with the same properties is equal'''
        return (hasattr(another, 'movement') and self.movement == another.movement) \
            and (hasattr(another, 'rotation') and self.rotation == another.rotation) 

    def __hash__(self):
        '''Override hashing such that different instances can give the same key'''
        return hash((self.movement,self.rotation))

    def __repr__(self):
        '''Override to allow easier printing of actions'''
        return 'action' + str((self.movement, self.rotation))

class mdp(object):
    '''General Markov Decision Process class'''
    def __init__(self, S, A, transition_prob_dict, reward_function, gamma):
        '''Constructor

        Arguments:
        S - set of states: state class listed above. All possible states of the MDP.
        A - set of actions: action class listed above. All possible actions of the MDP.
        transition_prob_dict - dict of dicts of floats: The top level is hashed by 
            (state, action) pairs while the bottom level is hashed by future states, s_p. 
            The float values are the possible probabilities of moving to s_p from the
            specified state through the specified action.
        reward_function - function: The function should take in a state and return a float.
            The function is bounded to the instance of this class.
        gamma - float: Discount value that lowers future rewards.
        '''

        # maximum allowed policy optimization iterations
        # Note: this was placed to keep optimization processes finite. Remove or make larger
        # for larger systems
        self.MAX_ITERATIONS = 1000

        # tolerance for value differences. Needed to reduce optimization times
        self.V_TOLERANCE = 1e-3

        # set instance variables
        self.S = S
        self.A = A
        self.transition_prob_dict = transition_prob_dict
        self.reward_function = reward_function
        self.gamma = gamma

    def getTransitionProbability(self, s, a, s_p):
        '''Finds the probability for the system to transition from a state, s, to a new state,
        s_p, through a specified action, a. 
        
        Arguments:
        s - state: state class listed above. Current state that the action will applied on.
        a - action: action class listed above. Action that will be applied on to s.
        s_p - state: state class listed above. Proposed new state.
        '''
        return self.transition_prob_dict[(s,a)][s_p]

    def getPossibleNewStates(self, s,a):
        '''Finds the possible new states that the system could transition to from state, s,
        through action, a.
        
        Arguments:
        s - state: state class listed above. Current state that the action will applied on.
        a - action: action class listed above. Action that will be applied on to s.
        '''
        return self.transition_prob_dict[(s,a)].keys()

    def getNewState(self, s, a):
        '''Randomly returns a new state based on the transition probabilities of the MDP.

        Arguments:
        s - state: state class listed above. Current state that the action will applied on.
        a - action: action class listed above. Action that will be applied on to s.
        '''

        # find possible new states and their probabilities
        possible_new_states = self.getPossibleNewStates(s,a)
        possible_probabilities = [self.getTransitionProbability(s,a,s_p) for s_p in possible_new_states]

        # divide the range (0,1] into partitions that are proportionally sized by the transition
        # probabilities of the possible states
        thresholds = np.insert(np.cumsum(possible_probabilities),0,0)
        
        # choose random uniform value
        random_value = np.random.rand()

        # find where the random value falls and return that new state
        for k in range(len(thresholds)-1):
            if random_value > thresholds[k] and random_value < thresholds[k+1]:
                return possible_new_states[k]

    def getReward(self, s):
        '''Returns the reward for a state based on the MDP reward function.

        Arguments:
            s - state: state action class listed above.
        '''
        return self.reward_function(s)

    def runPolicyEvaluation(self, policy):
        '''Runs policy evalution on a provided policy and returns a dict with the values associated 
        with each state.

        Arguments:
            policy - dict of actions: The dict is hashed by state and has values of actions according
                to the current provided policy.
        '''

        # initialize a dict with 0 values
        # Note: defaultdict was used as it allows to call for any state that doesn't currently have
        # a value. Eventually, all valid keys will fill up with values. This might cause issues as
        # it returns 0 values for any state that may not be in the state space.
        V = defaultdict(lambda: 0, {})
        V_old = copy.copy(V)

        # initialize counters
        iterations = 0
        value_changed = True

        # run as long as the values have not converged
        while value_changed:
            value_changed = False

            # iterately calculate value by propagating rewards through one step lookahead 
            for s in self.S:

                # find the possible new states from the policy
                a = policy[s]
                possible_new_states = self.getPossibleNewStates(s,a)

                # calculate a new value 
                v_sum = 0
                for s_p in possible_new_states:
                    v_sum = v_sum + self.getTransitionProbability(s, a, s_p)*(self.getReward(s) + self.gamma*V_old[s_p])
                
                # check if the value was large enough from the current value
                if np.abs(V[s] - v_sum) > self.V_TOLERANCE:
                    value_changed = True

                # update the value
                V[s] = v_sum

            iterations += 1
            V_old = copy.copy(V)

        return V

    def runPolicyIteration(self, initial_policy):
        '''Runs policy iteration on an initial policy until the policy converges and returns the optimal
        policy and values.

        Arguments:
            initial_policy - dict of actions: action class listed above. The initial policy can be theoretically
                initialized using any method.

        '''
        logger.info('Starting Policy Iteration')
        start_time = time.time()

        # create an anonymous function to maximize the sum of the next steps values
        f = lambda fs, fa, fs_p : self.getTransitionProbability(fs, fa, fs_p)*V[fs_p]

        # reset loop variables
        iterations = 0
        last_policy = None
        
        # shallow copy so that the initial policy is not affected
        policy = copy.copy(initial_policy)

        # run until policy converges
        while not (policy == last_policy) and iterations < self.MAX_ITERATIONS:

            # shallow copy to compare against
            last_policy = copy.copy(policy)
            iterations += 1

            # run the policy evaluation to find the values
            V = self.runPolicyEvaluation(policy)

            # run through state space and maximize the possible next step values
            for s in self.S:
                _, policy[s] = self.maximizeFunctionOverActions(f,s)

        run_time = time.time() - start_time
        logger.info('Finishing Policy Iteration. Runtime: %6.3f sec' % run_time)
        return policy, V

    def runValueIteration(self, initial_policy):
        logger.info('Starting Value Iteration')
        start_time = time.time()

        V = defaultdict(lambda: 0, {})
        V_old = copy.copy(V)

        value_changed = True
        iterations = 0

        f = lambda fs, fa, fs_p : self.getTransitionProbability(fs, fa, fs_p)*(self.getReward(fs) + self.gamma*V_old[fs_p])


        # policy = initializePolicy(state(GOAL_X,GOAL_Y,0))
        policy = copy.copy(initial_policy)
        while value_changed and iterations < self.MAX_ITERATIONS:
            value_changed = False
            iterations += 1

            for s in self.S:
                V[s], policy[s] = self.maximizeFunctionOverActions(f, s)

                if np.abs(V[s] - V_old[s]) > self.V_TOLERANCE:
                    value_changed = True

            V_old = copy.copy(V)

        run_time = time.time() - start_time
        logger.info('Finishing Value Iteration. Runtime: %6.3f sec' % run_time)
        return policy

    def maximizeFunctionOverActions(self, f, s):
        max_f_value = -1e10
        for a in self.A:
            f_value = 0
            possible_new_states = self.getPossibleNewStates(s,a)
            for s_p in possible_new_states:
                f_value += f(s, a, s_p)

            if f_value > max_f_value:
                max_f_value = f_value
                max_action = a

        return max_f_value, max_action


if __name__ == '__main__':
    # create example state
    s = state(1,2,2)

    # create example action
    a = action('forwards','right',np.array([[0.5], [np.sqrt(3)/2]]))

    print s, a