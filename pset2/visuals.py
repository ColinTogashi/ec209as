# TODO: script header

# TODO: add trajectory?
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

TIME_TO_PAUSE = 0.5

def convertHeadingToRadians(heading):
    return -np.pi*np.mod(heading-3,12)*30/180 + 2*np.pi

class gridWorld(object):

    def __init__(self, title, reward_matrix, goal_state):

        self.ARROW_LENGTH = 0.4
        self.ARROW_WIDTH = 0.3

        x_axis_size = reward_matrix.shape[1]
        y_axis_size = reward_matrix.shape[0]
        xticklabels = range(0, x_axis_size) # could be text
        yticklabels = range(0, y_axis_size) # could be text   

        cmap = mpl.colors.ListedColormap(['yellow', 'white'])
        cmap.set_over('green')
        cmap.set_under('red')

        bounds = np.unique(reward_matrix)[:-1] + 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()

        # Plot it out
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

        # Turn off all the ticks
        ax = plt.gca()    
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        plt.plot(goal_state.x+0.5, goal_state.y+0.5, marker='*', color='k', markersize=10)
        
        self.fig = fig
        self.ax = ax
        
        self.goal = goal_state

    def setStartState(self, start_state):
        plt.plot(start_state.x+0.5, start_state.y+0.5, marker='.', color='k', markersize=10)
        
        heading_radians = convertHeadingToRadians(start_state.heading)
        dx = self.ARROW_LENGTH*np.cos(heading_radians)
        dy = self.ARROW_LENGTH*np.sin(heading_radians)

        arrow = mpl.patches.Arrow(start_state.x+0.5, start_state.y+0.5, 
                                    dx, dy, width=self.ARROW_WIDTH)
        arrow.set_edgecolor('k')
        self.arrow = self.ax.add_patch(arrow)

        self.state = start_state

        plt.pause(TIME_TO_PAUSE)


    def updateState(self,s):

        if self.arrow is not None:
            self.arrow.remove()

            heading_radians = convertHeadingToRadians(s.heading)
            dx = self.ARROW_LENGTH*np.cos(heading_radians)
            dy = self.ARROW_LENGTH*np.sin(heading_radians)
            arrow = mpl.patches.Arrow(s.x+0.5, s.y+0.5, dx, dy, width=self.ARROW_WIDTH)
            arrow.set_edgecolor('k')
            self.arrow = self.ax.add_patch(arrow)

            self.state = s

            plt.pause(TIME_TO_PAUSE)

            
        else:
            logger.error('Grid World: First call setStartState() before running updateState()')


if __name__ == '__main__':
    import numpy as np
    from common import state

    start_state = state(1,4,3)
    intermediate_state = state(2,4,2)
    goal_state = state(3,4,1)

    REWARD_MATRIX = np.zeros((6,6))
    REWARD_MATRIX[:,0] = -100
    REWARD_MATRIX[:,-1] = -100
    REWARD_MATRIX[0,:] = -100
    REWARD_MATRIX[-1,:] = -100
    REWARD_MATRIX[2:5,2] = -10
    REWARD_MATRIX[2:5,4] = -10
    REWARD_MATRIX[4,3] = 1

    grid_world = gridWorld('6x6 grid', REWARD_MATRIX, goal_state)

    grid_world.setStartState(start_state)
    plt.pause(1)
    grid_world.updateState(intermediate_state)
    plt.pause(1)
    grid_world.updateState(goal_state)
    plt.pause(1)