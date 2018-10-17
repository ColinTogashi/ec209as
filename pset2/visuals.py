#!/usr/bin/env python
'''Module containing all graphical objects and supporting functions.'''

__author__ = 'Colin Togashi'

# libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import os

# libraries for logging
from debug_logger import setupLogging
import logging

# path where images are stored
IMAGE_PATH = './images'

# setup logging for use in module
setupLogging()
logger = logging.getLogger(__name__)
logger.debug('Importing visuals.py')

def convertHeadingToRadians(heading):
    '''Converts heading (0->11) into radians (0->2*pi) 
    
    Arguments:
        heading - int: describes direction of robot with 0 as positive y direction and
            incrementing in the clockwise direction
    '''
    return -np.pi*np.mod(heading-3,12)*30/180 + 2*np.pi

class gridWorld(object):
    '''Graphics object to animate algorithms'''

    def __init__(self, title, reward_matrix, possible_goal_states):
        '''Constructor

        Arguments:
            title - string: title that is shown on top of grid
            reward_matrix - 2D numpy array: rewards that correspond to each grid position
            possible_goal_states - set of states: state defined in common.py 
                to denote all the possible goals states
         '''

        logger.debug('Creating a new gridWorld')

        # to show a gradient trajectory, choose blue and pink
        self.TRAJECTORY_START_COLOR = np.array([50, 38, 229])/255.0
        self.TRAJECTORY_END_COLOR = np.array([229, 9, 222])/255.0

        # constant for defining delay between frames
        self.PAUSE_DELAY = 0.5

        # constants for defining heading arrow properties
        self.ARROW_LENGTH = 0.1
        self.ARROW_WIDTH = 0.05

        # assign x and y ticks on the grid
        x_axis_size = reward_matrix.shape[1]
        y_axis_size = reward_matrix.shape[0]
        xticklabels = range(0, x_axis_size) # could be text
        yticklabels = range(0, y_axis_size) # could be text   

        # create a color map to match problem statement
        # Note: currently only supports 4 reward values. Move to a transition color map for more.
        cmap = mpl.colors.ListedColormap(['yellow', 'white'])
        cmap.set_over('green')
        cmap.set_under('red')

        # create discrete bounds to divide the values
        bounds = np.unique(reward_matrix)[:-1] + 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # needed to automatically update plot
        plt.ion()

        # plot out rewards
        fig, ax = plt.subplots()    
        c = ax.pcolor(reward_matrix, edgecolors='k', linestyle= 'dashed', 
                         linewidths=0.2, cmap=cmap, norm=norm)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(reward_matrix.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(reward_matrix.shape[1]) + 0.5, minor=False)

        # set tick labels
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')      

        # turn off all the ticks marks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # all goal states have the same x and y so choose the first one for display
        goal_state = next(iter(possible_goal_states))

        # plot a star at the goal state x, y
        plt.plot(goal_state.x+0.5, goal_state.y+0.5, marker='*', color='k', markersize=10)
        
        # save the figure, axes, and possible goal states for use in other methods
        self.fig = fig
        self.ax = ax
        self.possible_goal_states = possible_goal_states

        self.start_marker = None

        # resets key instance variables 
        self.resetVariables()

        # since interactive plotting is on, need to pause in order to see the initial plot
        plt.pause(self.PAUSE_DELAY)


    def resetVariables(self):
        '''Resets all instance variables to default values'''

        logger.debug('Resetting plot variables')

        # clear objects from plot
        if self.start_marker is not None:
            self.start_marker.remove()
            self.state_arrow.remove()
            self.trajectory_line.remove()

            for arrow in self.trajectory_arrows:
                arrow.remove()
        
        # reset graphics objects
        self.state_arrow = None
        self.trajectory_line = None
        self.start_marker = None
        self.trajectory_arrows = []
        
        # reset internal variables
        self.state_number = 0
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory = []
        

    def setStartState(self, start_state):
        '''Plot and initialize the starting state

        Arguments:
            start_state - state: state defined in common.py to denote the starting point
        '''

        # update the state to the start state.
        # Note: moved all start state code to updateState() and only to keep interfaces
        self.updateState(start_state)


    def updateState(self,s):
        '''Update plot to a new state and plot trajectory
        
        Arguments:
            s - state: state defined in common.py to denote the current state
        '''

        logger.debug('Updating with new state: ' + str(s))

        # add points of the trajectory
        self.trajectory_x.append(s.x+0.5)
        self.trajectory_y.append(s.y+0.5)
        self.trajectory.append(s)

        # check if the starting state has been specified already
        if self.start_marker is None:
            # plot a marker to denote the starting state
            self.start_marker, =  plt.plot(s.x+0.5, s.y+0.5, marker='.', color='k', markersize=10)

            # plot trajectory from the previous state to this state
            self.trajectory_line, = plt.plot(self.trajectory_x, self.trajectory_y, 'k--')    
        else:
            # since arrow updates aren't built in to matplotlib, delete arrow to recreate later
            self.state_arrow.remove()

            # update trajectory with new values
            self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)
            
        # calculate the heading in radians to point the arrow
        heading_radians = convertHeadingToRadians(s.heading)

        # calculate the change in x and y to show heading with the arrow object
        dx = self.ARROW_LENGTH*np.cos(heading_radians)
        dy = self.ARROW_LENGTH*np.sin(heading_radians)

        # create a new arrow pointing in the correct heading
        # Note: store arrow to destroy and recreate later
        self.state_arrow = self.ax.arrow(s.x+0.5, s.y+0.5, dx, dy, width=self.ARROW_WIDTH, edgecolor='k')

        # pause to see the plot update
        plt.pause(self.PAUSE_DELAY)

        # save the current state for future calculations
        self.state = s
        self.state_number += 1


    def runSimulation(self, getNewState, policy, start_state, image_base_name=None, image_format='.pdf'):
        '''Given a system, policy, and starting state, animates the simulation to the end

        Arguments:
            getNewState - function: this function describes how a new state is found in the system
            policy - dict of actions: the given policy must have actions that lead to the goal
            start_state - state: state defined in common.py to denote the start point
        '''

        logger.info('Starting a new simulation')

        # reset instance variables
        self.resetVariables()

        # update plot with the starting state
        self.updateState(start_state)
        if image_base_name is not None:
            image_name = image_base_name + 'Start'
            self.saveFigure(image_name, image_format)

        # change the current state
        current_state = start_state

        # run simulation while the current state is not the goal state
        # TODO: should there also be a condition for bad policies that will never find a way?
        while not (current_state in self.possible_goal_states):
            # get the current action and find a new state based on the system
            current_action = policy[current_state]
            new_state = getNewState(current_state, current_action)

            # update plot with the new state
            self.updateState(new_state)
            if image_base_name is not None:
                image_name = image_base_name + str(self.state_number)
                self.saveFigure(image_name, image_format)

            # update current state
            current_state = new_state

        self.plotTrajectoryGradient()

        # increment state number to allow saving a new figure
        if image_base_name is not None:
            image_name = image_base_name + 'End'
            self.saveFigure(image_name, image_format)

        logger.info('Finishing simluation')

    def plotTrajectoryGradient(self):
        '''Plots the state trajectory with gradient colors to show time progression'''

        # total number of states in the trajectory is the current state number
        num_trajectory_states = self.state_number

        # linearly interpolate all color channels by the number of states
        red = np.linspace(self.TRAJECTORY_START_COLOR[0],self.TRAJECTORY_END_COLOR[0],num_trajectory_states);
        green = np.linspace(self.TRAJECTORY_START_COLOR[1],self.TRAJECTORY_END_COLOR[1],num_trajectory_states);
        blue = np.linspace(self.TRAJECTORY_START_COLOR[2],self.TRAJECTORY_END_COLOR[2],num_trajectory_states);
        
        # create a gradient color set
        gradient_color_set = np.array([red, green, blue]) 

        # plot an arrow 
        for k in range(num_trajectory_states):
            # pull state from trajectory
            s = self.trajectory[k]

            # calculate the heading in radians to point the arrow
            heading_radians = convertHeadingToRadians(s.heading)

            # calculate the change in x and y to show heading with the arrow object
            dx = 0.3*np.cos(heading_radians)
            dy = 0.3*np.sin(heading_radians)

            # plot an arrow denoting part of the trajectory
            self.trajectory_arrows.append(self.ax.arrow(s.x+0.5, s.y+0.5, dx, dy, width=self.ARROW_WIDTH, 
                facecolor=(gradient_color_set[:,k]), edgeColor=(gradient_color_set[:,k])))

    def saveFigure(self, image_name, image_format):
        '''Modified saving to allow for filename and format specific names'''

        filename = os.path.join(IMAGE_PATH,image_name+image_format)
        logger.info('Saving image: %s' % filename)
        self.fig.savefig(filename)


if __name__ == '__main__':
    # TODO: put in how to run a simulation
    import numpy as np
    from common import state

    # make dummy states
    start_state = state(1,4,3)
    second_state = state(2,4,2)
    third_state = state(3,4,1)
    possible_goal_states = [state(3,4,11), state(3,4,0), state(3,4,1)]

    # make dummy rewards matrix
    REWARD_MATRIX = np.zeros((6,6))
    REWARD_MATRIX[:,0] = -100
    REWARD_MATRIX[:,-1] = -100
    REWARD_MATRIX[0,:] = -100
    REWARD_MATRIX[-1,:] = -100
    REWARD_MATRIX[2:5,2] = -10
    REWARD_MATRIX[2:5,4] = -10
    REWARD_MATRIX[4,3] = 1

    # create grid world object
    grid_world = gridWorld('6x6 grid', REWARD_MATRIX, possible_goal_states)

    # run through a quick simulation of using the grid world
    # Note: updateState should be called to initialize as startState() is now obsolete
    grid_world.updateState(start_state)
    grid_world.updateState(second_state)
    grid_world.updateState(third_state)
    grid_world.plotTrajectoryGradient()

    raw_input('Press Enter when finished')