import numpy as np
import copy
import time

from collections import defaultdict

from debug_logger import *
from visuals import gridWorld
from common import state, action

# TODO: visuals to plot trajectory
# TODO: clean up code so that there isn't so much outside functions
# TODO: be careful about moving GOAL_X and GOAL_Y since they are now part of initializePolicy()
# TODO: should initializePolicy take in goal state?



# dimensions of the grid world
L = 6
W = 6
NUM_HEADINGS = 12

# pre-rotation error probability
error_probability = 0.0
gamma = 0.9

V_TOLERANCE = 1e-3
MAX_ITERATIONS = 100

# create constant rewards matrix based on problem
# matrix organized relative to coordinate system
# ie. to access reward given for x = 3, y = 2 => REWARD_MATRIX[3,2]
REWARD_MATRIX = np.zeros((W,L,NUM_HEADINGS))
REWARD_MATRIX[:,0,:] = -100
REWARD_MATRIX[:,-1,:] = -100
REWARD_MATRIX[0,:,:] = -100
REWARD_MATRIX[-1,:,:] = -100
REWARD_MATRIX[2,2:5,:] = -10
REWARD_MATRIX[4,2:5,:] = -10
REWARD_MATRIX[3,4,5:8] = 1

REWARD_MATRIX_DISPLAY = np.zeros((W,L))
REWARD_MATRIX_DISPLAY[:,0] = -2
REWARD_MATRIX_DISPLAY[:,-1] = -2
REWARD_MATRIX_DISPLAY[0,:] = -2
REWARD_MATRIX_DISPLAY[-1,:] = -2
REWARD_MATRIX_DISPLAY[2:5,2] = -1
REWARD_MATRIX_DISPLAY[2:5,4] = -1
REWARD_MATRIX_DISPLAY[4,3] = 1


# create lists of all possible position and heading values
x_values = np.arange(0,L)
y_values = np.arange(0,W)
heading_values = np.arange(0,NUM_HEADINGS)

# size of the state space
Ns = L*W*NUM_HEADINGS

# create state space by pulling each combination of x, y, and heading values 
S = {state(x,y,heading) for x in x_values for y in y_values for heading in heading_values}

# create set of all actions
VECTOR_TOLERANCE = 0.001

A = set()
A.add(action('forwards',None,np.array([[0], [1]])))
A.add(action('forwards','right',np.array([[0.5], [np.sqrt(3)/2]])))
A.add(action('forwards','left',np.array([[-0.5], [np.sqrt(3)/2]])))
A.add(action('backwards',None,np.array([[0], [-1]])))
A.add(action('backwards','right',np.array([[0.5-VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
A.add(action('backwards','left',np.array([[-0.5+VECTOR_TOLERANCE], [-np.sqrt(3)/2]])))
A.add(action('stay',None,np.array([[0], [0]])))


# given a state and an action, return a new state based on the grid
def tryToMove(s,a):
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


TRANSITION_PROBABILITY_TABLE = {}
for s in S:
    for a in A:
        TRANSITION_PROBABILITY_TABLE[(s,a)] = {}

        if a.movement == 'stay':
            new_s = s
            TRANSITION_PROBABILITY_TABLE[(s,a)][new_s] = 1
        elif a.movement == 'forwards' or a.movement == 'backwards':
            left_rotated_s = copy.copy(s)
            right_rotated_s = copy.copy(s)
            left_rotated_s.heading = (left_rotated_s.heading - 1) % NUM_HEADINGS
            right_rotated_s.heading = (right_rotated_s.heading + 1) % NUM_HEADINGS

            new_s = tryToMove(s,a)
            left_rotated_new_s = tryToMove(left_rotated_s,a)
            right_rotated_new_s = tryToMove(right_rotated_s,a)                

            TRANSITION_PROBABILITY_TABLE[(s,a)][left_rotated_new_s] = error_probability
            TRANSITION_PROBABILITY_TABLE[(s,a)][right_rotated_new_s] = error_probability
            TRANSITION_PROBABILITY_TABLE[(s,a)][new_s] = 1 - 2*error_probability

        else:
            logger.error('The action is not supported.')


def transitionProbability(error_probability, s, a, new_s):
    ''' This function finds the probability for the robot to transition from a state
    to a new state through a specified action. '''
    return TRANSITION_PROBABILITY_TABLE[(s,a)][new_s]


def findNewState(s, a):
    possible_new_states = TRANSITION_PROBABILITY_TABLE[(s,a)].keys()
    possible_probabilities = TRANSITION_PROBABILITY_TABLE[(s,a)].values()
    thresholds = np.insert(np.cumsum(possible_probabilities),0,0)
    random_value = np.random.rand()

    for k in range(len(thresholds)-1):
        if random_value > thresholds[k] and random_value < thresholds[k+1]:
            return possible_new_states[k]


def returnReward(state):
    # TODO: Set up a way to easily switch between modified reward function that is heading dependent
    return REWARD_MATRIX[state.x,state.y,state.heading]




def initializePolicy(goal_state):
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


title = '6 x 6 Grid World'

GOAL_X = 3
GOAL_Y = 4

START_X = 1
START_Y = 4
START_HEAD = 6

POSSIBLE_GOAL_STATES = {state(GOAL_X, GOAL_Y, heading) for heading in heading_values}

goal_state = state(3,4,0)
start_state = state(1,4,6)
grid_world = gridWorld(title, REWARD_MATRIX_DISPLAY, goal_state)
grid_world.setStartState(start_state)

def runSimulation(grid_world, policy, start_state):
    current_state = start_state

    while not (current_state in POSSIBLE_GOAL_STATES):
        current_action = policy[current_state]
        new_state = findNewState(current_state, current_action)
        grid_world.updateState(new_state)
        current_state = new_state



def runPolicyEvaluation(policy):
    V = defaultdict(lambda: 0, {})
    V_old = copy.copy(V)

    iterations = 0
    value_changed = True

    while value_changed:
        value_changed = False
        for s in S:
            a = policy[s]
            possible_new_states = TRANSITION_PROBABILITY_TABLE[(s,a)].keys()

            v_sum = 0
            for s_p in possible_new_states:
                v_sum = v_sum + TRANSITION_PROBABILITY_TABLE[(s,a)][s_p]*(returnReward(s) + gamma*V_old[s_p])
            
            if np.abs(V[s] - v_sum) > V_TOLERANCE:
                value_changed = True

            V[s] = v_sum

        iterations += 1
        V_old = copy.copy(V)

    return V

def runPolicyIteration():
    start_time = time.time()

    iterations = 0

    f = lambda fs, fa, fs_p : TRANSITION_PROBABILITY_TABLE[(fs,fa)][fs_p]*V[fs_p]

    last_policy = None
    policy = initializePolicy(state(GOAL_X,GOAL_Y,0))
    while not (policy == last_policy) and iterations < MAX_ITERATIONS:
        last_policy = copy.copy(policy)
        iterations += 1
        V = runPolicyEvaluation(policy)
        for s in S:
            _, policy[s] = maximizeFunctionOverActions(f,s)

    run_time = time.time() - start_time
    return policy



def runValueIteration():
    start_time = time.time()

    V = defaultdict(lambda: 0, {})
    V_old = copy.copy(V)

    value_changed = True
    iterations = 0

    f = lambda fs, fa, fs_p : TRANSITION_PROBABILITY_TABLE[(fs,fa)][fs_p]*(returnReward(fs) + gamma*V_old[fs_p])


    policy = initializePolicy(state(GOAL_X,GOAL_Y,0))
    while value_changed and iterations < MAX_ITERATIONS:
        value_changed = False
        iterations += 1

        for s in S:
            V[s], policy[s] = maximizeFunctionOverActions(f, s)

            if np.abs(V[s] - V_old[s]) > V_TOLERANCE:
                value_changed = True

        V_old = copy.copy(V)

    run_time = time.time() - start_time
    return policy


def maximizeFunctionOverActions(f, s):
    max_f_value = -1e10
    for a in A:
        f_value = 0
        possible_new_states = TRANSITION_PROBABILITY_TABLE[(s,a)].keys()
        for s_p in possible_new_states:
            f_value += f(s, a, s_p)

        if f_value > max_f_value:
            max_f_value = f_value
            max_action = a

    return max_f_value, max_action

p = runPolicyIteration()

runSimulation(grid_world, p, start_state)