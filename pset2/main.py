#!/usr/bin/env python

import numpy as np
import copy
import time

from collections import defaultdict

from debug_logger import setupLogging
import logging
from visuals import gridWorld
from common import state, action, mdp

setupLogging()
logger = logging.getLogger(__name__)

logger.info('Start main program')

# dimensions of the grid world
L = 6
W = 6
NUM_HEADINGS = 12

# define the goal position
GOAL_X = 3
GOAL_Y = 4

# define the start position
START_X = 1
START_Y = 4
START_HEAD = 6

# create constant rewards matrix based on problem
# matrix organized relative to coordinate system
# ie. to access reward given for x = 3, y = 2 => reward_matrix[3,2]
def createRewardMatrix(L, W, num_headings):
    '''Creates a rewards matrix based on the problem statement. This matrix is organized relative
    to the coordinate system. ie. to access reward given for x = 3, y = 2 => reward_matrix[3,2,heading]

    Arguments:
        L - int: length of grid
        W - int: width of grid
        num_headings - int: number of headings to consider
    '''

    reward_matrix = np.zeros((W,L,NUM_HEADINGS))
    reward_matrix[:,0,:] = -100
    reward_matrix[:,-1,:] = -100
    reward_matrix[0,:,:] = -100
    reward_matrix[-1,:,:] = -100
    reward_matrix[2,2:5,:] = -6
    reward_matrix[4,2:5,:] = -6
    reward_matrix[3,4,5:8] = 25

    # reward_matrix[3,4,0:2] = 1
    # reward_matrix[3,4,-1] = 1
    return reward_matrix


def createDisplayRewardMatrix(L, W, num_headings):
    '''Creates a rewards matrix based on the problem statement. This matrix is organized relative
    to the rows and columns. ie. to access reward given for r = 2, c = 3 => reward_matrix[2,3,heading]

    Arguments:
        L - int: length of grid
        W - int: width of grid
        num_headings - int: number of headings to consider
    '''
    display_reward_matrix = np.zeros((W,L))
    display_reward_matrix[:,0] = -100
    display_reward_matrix[:,-1] = -100
    display_reward_matrix[0,:] = -100
    display_reward_matrix[-1,:] = -100
    display_reward_matrix[2:5,2] = -10
    display_reward_matrix[2:5,4] = -10
    display_reward_matrix[4,3] = 10

    return display_reward_matrix


def runDynamics(s,a):
    '''Given a state and action, returns a new state based on the defined dynamics of the robot

    Arguments:
        s - state: state class defined in common.py. Current specified state
        a - action: action class defined in common.py. Current specified action
    '''

    # shallow copy to base calculations on without disturbing original state
    new_state = copy.copy(s)

    # change modifier based on whether the robot moves forwards or backwards
    if a.movement == 'forwards':
        move_modifier = 1
    elif a.movement == 'backwards':
        move_modifier = -1
    else:
        logger.error('Movement incorrectly specified in the action.')
        pass

    # based on the heading, compute which direction the robot moves in
    if s.heading >= 2 and s.heading  <= 4:
        new_state.x += move_modifier
    elif s.heading  >= 5 and s.heading  <= 7:
        new_state.y -= move_modifier
    elif s.heading  >= 8 and s.heading  <= 10: 
        new_state.x -= move_modifier
    elif s.heading  == 11 or s.heading  == 0 or s.heading  == 1:
        new_state.y += move_modifier
    else:
        print s.heading
        logger.error('Heading incorrectly specified in the state.')

    # check the bounds such that the robot always stays in the state space
    if new_state.x > L-1:
        new_state.x = L-1
    if new_state.y > W-1:
        new_state.y = W-1
    if new_state.x < 0:
        new_state.x = 0
    if new_state.y < 0:
        new_state.y = 0

    # compute the new heading
    if a.rotation == 'right':
        new_state.heading = (new_state.heading + 1) % NUM_HEADINGS
    elif a.rotation == 'left':
        new_state.heading = (new_state.heading - 1) % NUM_HEADINGS
    elif a.rotation == None:
        pass
    else:
        logger.error('Rotation incorrectly specified in action.')

    return new_state


def createTransitionProbabilityTable(S,A,error_probability):
    '''Creates a transition probability table that contains all the system information on how the
    robot should behave given the dynamics. The table is hashed by a (state, action) pair and then
    by a new_state

    Arguments:
        S - set of states: state class defined in common.py. Represents all states
        A - set of actions: action class defined in common.py. Represents all actions
        error_probability - float: the probability that an a pre-rotation will occur
    '''

    transition_probability_dict = {}
    
    # loop over state space
    for s in S:
        # loop over action space
        for a in A:

            transition_probability_dict[(s,a)] = {}
            
            # if a robot stays, it will always end up in the same state
            if a.movement == 'stay':    
                s_p = s
                transition_probability_dict[(s,a)][s_p] = 1

            # if a robot moves, it can pre-rotate before undertaking an action
            elif a.movement == 'forwards' or a.movement == 'backwards':

                # shallow copy the state to propogate pre-rotation errors
                left_rotated_s = copy.copy(s)
                right_rotated_s = copy.copy(s)

                # simulate pre-rotation errors
                left_rotated_s.heading = (left_rotated_s.heading - 1) % NUM_HEADINGS
                right_rotated_s.heading = (right_rotated_s.heading + 1) % NUM_HEADINGS

                # run the dynamics on the error and non-error states
                s_p = runDynamics(s,a)
                left_rotated_s_p = runDynamics(left_rotated_s,a)
                right_rotated_s_p = runDynamics(right_rotated_s,a)                

                # assign probabilities corresponding to the new states
                transition_probability_dict[(s,a)][left_rotated_s_p] = error_probability
                transition_probability_dict[(s,a)][right_rotated_s_p] = error_probability
                transition_probability_dict[(s,a)][s_p] = 1 - 2*error_probability

            else:
                logger.error('The action is not supported.')
    return transition_probability_dict


def initializePolicy(S, A, goal_state):
    '''Initializes policy as the action that will point/move the robot closer to the goal x, y position.
    Computes the direction of the goal state relative to the current position and finds the action with
    a direction vector that best aligns with that vector.

    Arguments:
        S - set of states: state class defined in common.py. Represents all states
        A - set of actions: action class defined in common.py. Represents all actions
        goal_state: action clas
    '''

    pi = {}
    # loop through state space
    for s in S:

        # find rotation matrices based on heading to modify action direction vectors
        if s.heading >= 2 and s.heading  <= 4:
            Rt = np.array([[0, 1],[-1, 0]])
        elif s.heading  >= 5 and s.heading  <= 7:
            Rt = np.array([[-1, 0],[0, -1]])
        elif s.heading  >= 8 and s.heading  <= 10: 
            Rt = np.array([[0, -1],[1, 0]])
        elif s.heading  == 11 or s.heading  == 0 or s.heading  == 1:
            Rt = np.array([[1, 0],[0, 1]])
        else:
            print s.heading
            logger.error('Heading incorrectly specified in the state.')

        # initiliaze max finding variables
        initial_action = None
        max_dot_product = -1e10
  
        # the starting state has the goal position and it does not need to move
        if (s.x == goal_state.x and s.y == goal_state.y):
            initial_action = action('stay',None,np.array([[0],[0]]))
        else:
            # loop over all actions
            for a in A:
                # determine the vector from the current state to the goal state
                state_to_goal_vector = np.array([[goal_state.x-s.x],[goal_state.y-s.y]])

                # rotate the normalized action direction vectors based on the current heading
                action_direction_vector = np.dot(Rt,a.vector)

                # compute the dot product and find the action that most aligns with the direction we want
                direction_dot_product = np.double(np.dot(state_to_goal_vector.T,action_direction_vector)) 
                if direction_dot_product > max_dot_product:
                    max_dot_product = direction_dot_product
                    initial_action = a

        # assign the best action to the policy
        pi[s] = initial_action

    return pi


def rewardFunctionNonHeadingDependent(reward_matrix, s):
    '''Computes the reward function that does not take into account the heading'''
    return np.max(reward_matrix[s.x,s.y,:])


def rewardFunctionHeadingDependent(reward_matrix, s):
    '''Computes the reward function that accounts for the heading'''
    return reward_matrix[s.x,s.y,s.heading]

def runProblem(error_probability, gamma, heading_dependent):
    '''Runs policy iteration and value iteration for each case'''

    # create lists of all possible position and heading values
    x_values = np.arange(0,L)
    y_values = np.arange(0,W)
    heading_values = np.arange(0,NUM_HEADINGS)
    down_heading_values = np.array([5,6,7])

    # size of the state space
    Ns = L*W*NUM_HEADINGS

    # create state space by pulling each combination of x, y, and heading values 
    S = {state(x,y,heading) for x in x_values for y in y_values for heading in heading_values}

    # create the action space 
    # action direction vectors are defined where a rotation is a 30 deg turn and forwards is in the positive
    # y direction
    # Note: since we must always go forward when the goal is directly to our right, we slightly adjust the
    # direction vector such that any 90 right or left turns will not align with a backwards movement
    VECTOR_TOLERANCE = 0.001

    A = set()
    A.add(action('forwards',None,np.array([[0], [1]])))
    A.add(action('forwards','right',np.array([[0.5], [np.sqrt(3)/2]])))
    A.add(action('forwards','left',np.array([[-0.5], [np.sqrt(3)/2]])))
    A.add(action('backwards',None,np.array([[0], [-1]])))
    A.add(action('backwards','right',np.array([[0.5-VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
    A.add(action('backwards','left',np.array([[-0.5+VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
    A.add(action('stay',None,np.array([[0], [0]])))


    # standard title
    title = '6 x 6 Grid World'

    if heading_dependent:
        # create new possible goal states
        possible_goal_states = {state(GOAL_X, GOAL_Y, heading) for heading in down_heading_values}
        # create a reward function to bind to the mdp
        reward_function = lambda s: rewardFunctionHeadingDependent(reward_matrix, s)

        head_string = '_down'
    else:
        # compute the possible goal states for all heading values
        possible_goal_states = {state(GOAL_X, GOAL_Y, heading) for heading in heading_values}
        # create a reward function to bind to the mdp
        reward_function = lambda s: rewardFunctionNonHeadingDependent(reward_matrix, s)

        head_string = ''

    # create a reward matrix for algorithm use and another to display
    reward_matrix = createRewardMatrix(L, W, NUM_HEADINGS)
    display_reward_matrix = createDisplayRewardMatrix(L, W, NUM_HEADINGS)

    # define goal states and start states
    goal_state = state(GOAL_X,GOAL_Y,0)
    start_state = state(START_X,START_Y,START_HEAD)

    # define the grid world
    grid_world = gridWorld(title, display_reward_matrix, possible_goal_states)

    # create the transition probability table
    transition_probability_dict = createTransitionProbabilityTable(S, A, error_probability)

    # initialize the policy
    initial_policy = initializePolicy(S, A, goal_state)

    # make a new mdp
    mdp_problem = mdp(S, A, transition_probability_dict, reward_function, gamma, possible_goal_states)

    # map the mdp new state function to the simulation
    getNewState = lambda s, a: mdp_problem.getNewState(s,a)

    # compute the initial value
    initial_value = mdp_problem.runPolicyEvaluation(initial_policy)
    logger.info('Value of initial state for initial policy: %f' % initial_value[start_state])

    if not heading_dependent:
        raw_input('Press Enter to show the initial policy')
        
        # simulate initial policy
        grid_world.runSimulation(getNewState, initial_policy, initial_value, start_state, 
                'initial_policy' + str(head_string) + '_pe' + str(error_probability) + '_')
    

    raw_input('Press Enter to show the Policy Iteration')

    # simulate policy iteration
    policy_iteration_policy, policy_iteration_value = mdp_problem.runPolicyIteration(initial_policy)
    logger.info('Value of initial state for initial policy: %f' % policy_iteration_value[start_state])
    grid_world.runSimulation(getNewState, policy_iteration_policy, policy_iteration_value, start_state, 
            'policy_iteration' + str(head_string) + '_pe' + str(error_probability) + '_')
    

    raw_input('Press Enter to show the Value Iteration')
    
    # simulate the value iteration
    value_iteration_policy, value_iteration_value = mdp_problem.runValueIteration(initial_policy)
    logger.info('Value of initial state for initial policy: %f' % value_iteration_value[start_state])
    grid_world.runSimulation(getNewState, value_iteration_policy, value_iteration_value, start_state, 
            'value_iteration' + str(head_string) + '_pe' + str(error_probability) + '_')
    
    raw_input('Press Enter to continue')



# discount factor
gamma = 0.95
# default settings
runProblem(0,gamma,False)

# error probability = 0.25
runProblem(0.25,gamma,False)

# heading dependent
runProblem(0,gamma,True)

# heading dependent, error probability = 0.25
runProblem(0.25,gamma,True)