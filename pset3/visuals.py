import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
L = 0.75
W = 0.5

def convertBearingToRadians(heading):
    '''Converts heading (0->360) into radians (0->2*pi) 
    
    Arguments:
        heading - int: describes direction of robot with 0 as positive y direction and
            incrementing in the clockwise direction
    '''
    return np.pi*np.mod(heading+90,360)/180


class robotWorld(object):
    def __init__(self, L, W):
        # constants for defining heading arrow properties
        self.ARROW_LENGTH = 0.02
        self.ARROW_WIDTH = 0.01

        self.L = L
        self.W = W

        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.axis([0, self.L, 0, self.W])

        self.fig.canvas.mpl_connect('key_press_event', self.press)

        self.robot_dot = None

        self.resetVariables()

    def resetVariables(self):

        if self.robot_dot is not None:
            self.robot_dot.remove()
            self.state_arrow.remove()

        self.robot_dot = None
        self.state_arrow = None
        self.st = None



    def updateState(self, st):
        if self.robot_dot is None:
            self.robot_dot, = plt.plot(st[1],st[2],'ro',markersize=14)

        else:
            self.state_arrow.remove()
            self.robot_dot.set_data(st[1],st[2])

        # calculate the change in x and y to show heading with the arrow object
        dx = self.ARROW_LENGTH*np.cos(st[0])
        dy = self.ARROW_LENGTH*np.sin(st[0])

        # create a new arrow pointing in the correct heading
        # Note: store arrow to destroy and recreate later
        self.state_arrow = self.ax.arrow(st[1], st[2], dx, dy, width=self.ARROW_WIDTH, edgecolor='k')


        self.st = st


    def press(self, event):
        # print event.key
        if event.key == 'up':
            self.st[2] += 0.05
        elif event.key == 'left':
            self.st[1] -= 0.05
        elif event.key == 'down':
            self.st[2] -= 0.05
        elif event.key == 'right':
            self.st[1] += 0.05

        self.updateState(self.st)

        if event.key == 'pageup':
            # right wheel forwards
            pass
        elif event.key == 'pagedown':
            # right wheel backwards
            pass
        elif event.key == 'insert':
            # left wheel forwards
            pass
        elif event.key == 'delete':
            # left wheel backwards    
            pass
    



S = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])
num_rows, num_cols = np.shape(S)
ax_list = []        
line_list = []
fig = plt.figure()
for r in range(num_rows):
    for c in range(num_cols):
        ax_list.append(plt.subplot(num_rows,num_cols,num_cols*r+c+1))
        line_list.append(plt.plot(S[r,c]))


if __name__ == '__main__':

    st = [0,0.2,0.2]
    robot_world = robotWorld(L,W)
    robot_world.updateState(st)
    input('Press Enter to End')