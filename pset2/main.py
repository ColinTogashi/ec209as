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

# TODO: clean up code so that there isn't so much outside functions

# dimensions of the grid world
L = 6
W = 6
NUM_HEADINGS = 12

# create constant rewards matrix based on problem
# matrix organized relative to coordinate system
# ie. to access reward given for x = 3, y = 2 => reward_matrix[3,2]
def createRewardMatrix(L, W, num_headings):

    reward_matrix = np.zeros((W,L,NUM_HEADINGS))
    reward_matrix[:,0,:] = -100
    reward_matrix[:,-1,:] = -100
    reward_matrix[0,:,:] = -100
    reward_matrix[-1,:,:] = -100
    reward_matrix[2,2:5,:] = -10
    reward_matrix[4,2:5,:] = -10
    reward_matrix[3,4,5:8] = 1

    # reward_matrix[3,4,0:2] = 1
    # reward_matrix[3,4,-1] = 1
    return reward_matrix

def createDisplayRewardMatrix(L, W, num_headings):
    display_reward_matrix = np.zeros((W,L))
    display_reward_matrix[:,0] = -2
    display_reward_matrix[:,-1] = -2
    display_reward_matrix[0,:] = -2
    display_reward_matrix[-1,:] = -2
    display_reward_matrix[2:5,2] = -1
    display_reward_matrix[2:5,4] = -1
    display_reward_matrix[4,3] = 1

    return display_reward_matrix

# create set of all actions
VECTOR_TOLERANCE = 0.001

# given a state and an action, return a new state based on the grid
def runDynamics(s,a):
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

    if new_state.x > L-1:
        new_state.x = L-1
    if new_state.y > W-1:
        new_state.y = W-1
    if new_state.x < 0:
        new_state.x = 0
    if new_state.y < 0:
        new_state.y = 0

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
    transition_probability_dict = {}
    for s in S:
        for a in A:
            transition_probability_dict[(s,a)] = {}

            if a.movement == 'stay':
                s_p = s
                transition_probability_dict[(s,a)][s_p] = 1
            elif a.movement == 'forwards' or a.movement == 'backwards':
                left_rotated_s = copy.copy(s)
                right_rotated_s = copy.copy(s)
                left_rotated_s.heading = (left_rotated_s.heading - 1) % NUM_HEADINGS
                right_rotated_s.heading = (right_rotated_s.heading + 1) % NUM_HEADINGS

                s_p = runDynamics(s,a)
                left_rotated_s_p = runDynamics(left_rotated_s,a)
                right_rotated_s_p = runDynamics(right_rotated_s,a)                

                transition_probability_dict[(s,a)][left_rotated_s_p] = error_probability
                transition_probability_dict[(s,a)][right_rotated_s_p] = error_probability
                transition_probability_dict[(s,a)][s_p] = 1 - 2*error_probability

            else:
                logger.error('The action is not supported.')
    return transition_probability_dict


def initializePolicy(S, A, goal_state):
    pi = {}
    for s in S:
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

        initial_action = None
        min_norm = 1e10
        max_dot_product = -1e10
  
        if (s.x == goal_state.x and s.y == goal_state.y):
            initial_action = action('stay',None,np.array([[0],[0]]))
        else:
            for a in A:
                state_to_goal_vector = np.array([[goal_state.x-s.x],[goal_state.y-s.y]])
                action_direction_vector = np.dot(Rt,a.vector)
                direction_dot_product = np.double(np.dot(state_to_goal_vector.T,action_direction_vector)) 
                if direction_dot_product > max_dot_product:
                    max_dot_product = direction_dot_product
                    initial_action = a

        pi[s] = initial_action

    return pi

def rewardFunctionNonHeadingDependent(reward_matrix, s):
    return np.max(reward_matrix[s.x,s.y,:])

def rewardFunctionHeadingDependent(reward_matrix, s):
    return reward_matrix[s.x,s.y,s.heading]


# pre-rotation error probability
error_probability = 0.0
gamma = 0.9

# create lists of all possible position and heading values
x_values = np.arange(0,L)
y_values = np.arange(0,W)
heading_values = np.arange(0,NUM_HEADINGS)

# size of the state space
Ns = L*W*NUM_HEADINGS

# create state space by pulling each combination of x, y, and heading values 
S = {state(x,y,heading) for x in x_values for y in y_values for heading in heading_values}



A = set()
A.add(action('forwards',None,np.array([[0], [1]])))
A.add(action('forwards','right',np.array([[0.5], [np.sqrt(3)/2]])))
A.add(action('forwards','left',np.array([[-0.5], [np.sqrt(3)/2]])))
A.add(action('backwards',None,np.array([[0], [-1]])))
A.add(action('backwards','right',np.array([[0.5-VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
A.add(action('backwards','left',np.array([[-0.5+VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
A.add(action('stay',None,np.array([[0], [0]])))

title = '6 x 6 Grid World'

GOAL_X = 3
GOAL_Y = 4

START_X = 1
START_Y = 4
START_HEAD = 6

POSSIBLE_GOAL_STATES = {state(GOAL_X, GOAL_Y, heading) for heading in heading_values}

reward_matrix = createRewardMatrix(L, W, NUM_HEADINGS)
display_reward_matrix = createDisplayRewardMatrix(L, W, NUM_HEADINGS)

goal_state = state(GOAL_X,GOAL_Y,0)
start_state = state(START_X,START_Y,START_HEAD)
grid_world = gridWorld(title, display_reward_matrix, POSSIBLE_GOAL_STATES)

transition_probability_dict = createTransitionProbabilityTable(S, A, error_probability)

initial_policy = initializePolicy(S, A, goal_state)

reward_function = lambda s: rewardFunctionNonHeadingDependent(reward_matrix, s)
mdp_reward_heading_independent = mdp(S, A, transition_probability_dict, reward_function, gamma)


policy_iteration_policy, policy_iteration_value = mdp_reward_heading_independent.runPolicyIteration(initial_policy)
value_iteration_policy, value_iteration_value = mdp_reward_heading_independent.runValueIteration(initial_policy)

getNewState = lambda s, a: mdp_reward_heading_independent.getNewState(s,a)

grid_world.runSimulation(getNewState, policy_iteration_policy, start_state, 'policy_iteration')

raw_input('Press Enter to continue')

grid_world.runSimulation(getNewState, value_iteration_policy, start_state, 'value_iteration')

raw_input('Press Enter to continue')
reward_function = lambda s: rewardFunctionHeadingDependent(reward_matrix, s)
mdp_reward_heading_dependent = mdp(S, A, transition_probability_dict, reward_function, gamma)