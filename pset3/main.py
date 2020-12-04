import numpy as np
from common import state
from debug_logger import setupLogging
import logging

# setup logging for use in module
setupLogging()
logger = logging.getLogger(__name__)
logger.debug('Starting Main Program')

WHEEL_RADIUS = 0.02 # [m]
WHEEL_SEPARATION_DISTANCE = 0.085 # [m]

L = 0.75 # [m]
W = 0.5 # [m]

def convertDecoupledInputsToCoupledInputs(ut):

    # Note: defined first input is left motor, second is right
    wt = WHEEL_RADIUS*(ut[1]-ut[0])/(2.0*WHEEL_SEPARATION_DISTANCE)
    vt = WHEEL_RADIUS*(ut[0]+ut[1])/2.0
    return wt, vt 

def nonlinearDynamics(st,u,dt):
    dt = 0.001
    wt, vt = convertDecoupledInputsToCoupledInputs(u)

    # TODO: fill in actual mean and sigma
    angular_noise = np.random.normal(0,1)
    velocity_noise = np.random.normal(0,1)

    thetat_plus_1  = st[0] + (wt + angular_noise)*dt
    xt_plus_1 = st[1] + (vt + velocity_noise)*np.cos(st[0])*dt
    yt_plus_1 = st[2] + (vt + velocity_noise)*np.sin(st[0])*dt

    st_plus_1 = [thetat_plus_1, xt_plus_1, yt_plus_1]
    return st_plus_1


# need to use takeMeasurement for true state to find distance
# also need to use takeMeasurement for to estimate landmark position based on state for Ht

def takeMeasurement(st):


    if np.abs(np.tan(st[0])) < 1e-9:
        distance_right = (L-st[1])/np.cos(st[0])
        distance_left = -st[1]/np.cos(st[0])

        y_right = st[2] + distance_right*np.sin(st[0])
        y_left = st[2] + distance_left*np.sin(st[0])

        y_list = [y_right,y_left]
        distance_list = [distance_right,distance_left]

        distance = np.max(distance_list)
        x_l = st[1]
        y_l = y_list[np.argmax(distance_list)]

    elif np.abs(np.tan(st[0])) > 1e9:
        distance_top = (W-st[2])/np.sin(st[0])
        distance_bot = (-st[2])/np.sin(st[0])

        x_top = st[1] + distance_top*np.cos(st[0])
        x_bot = st[1] + distance_bot*np.cos(st[0])

        x_list = [x_top,x_bot]
        distance_list = [distance_top,distance_bot]

        distance = np.max(distance_list)
        x_l = x_list[np.argmax(distance_list)]
        y_l = st[2]

    else:


        distance_right = (L-st[1])/np.cos(st[0])
        distance_left = -st[1]/np.cos(st[0])
        distance_top = (W-st[2])/np.sin(st[0])
        distance_bot = (-st[2])/np.sin(st[0])

        x_top = st[1] + distance_top*np.cos(st[0])
        x_bot = st[1] + distance_bot*np.cos(st[0])
        y_right = st[2] + distance_right*np.sin(st[0])
        y_left = st[2] + distance_left*np.sin(st[0])


        distance_list = [distance_right,distance_left,distance_top,distance_bot]
        x_list = [L,0,x_top,x_bot]
        y_list = [y_right,y_left,W,0]

        distance = np.sort(distance_list)[-2]
        x_l = x_list[np.argsort(distance_list)[-2]]
        y_l = y_list[np.argsort(distance_list)[-2]]

    return distance, (x_l, y_l)



class extendedKalmanFilter(object):
    def __init__(self, Ft_function, Wt_function, Ht_function, dt):
        

        self.Ft = lambda st, ut: Wt_function(st,ut,self.dt) # function of state, input, dt
        self.Wt = lambda st: Wt_function(st,self.dt) # function of state, dt
        self.Ht = Ht_function # function of state
        self.dt = dt

        self.st = None
        self.S = None
        self.S_inv = None
        self.Q = None
        self.R = None
        self.R_inv = None

    def initialize(self, st, S, Q, R):
        logger.debug('Initializing Extended Kalman Filter')
        self.st = st
        self.S = S
        self.S_inv = np.linalg.inv(S)
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)


    def timeUpdate(self, ut):
        logger.debug('Calling timeUpdate in Extended Kalman Filter')
        if self.st is not None:
            Ft = self.Ft(self.st,ut)
            Wt = self.Wt(self.st)
            self.st = Ft*self.st
            self.S = Ft*self.S*Ft.T + Wt*self.Q*Wt.T 
            self.S_inv = np.linalg.inv(self.S)
        else:
            logger.error('Calling timeUpdate before initializing!!!')

        return self.st, self.S

    def measurementUpdate(self, yt):
        if self.st is not None:
            logger.debug('Calling measurementUpdate in Extended Kalman Filter')
            Ht = self.Ht(self.st)
            # Vt = self.Vt(self.st)
            Kt = self.S*Ht.T*np.linalg.inv(Ht*self.S*Ht.T+self.R)
            self.st += Kt*(yt - Ht*self.st)
            self.S_inv = self.S_inv + Ht.T*self.R_inv*Ht
            self.S = np.linalg.inv(self.S_inv)
        else:
            logger.error('Calling measurementUpdate before initializing!!!')

        return self.st, self.S


def Ft_function(st, ut, dt):
    wt, vt = convertDecoupledInputsToCoupledInputs(ut)
    Ft = np.array([[                   1, 0, 0, 0],
                   [-vt*np.sin(st[0])*dt, 1, 0, 0],
                   [ vt*np.cos(st[0])*dt, 0, 1, 0],
                   [                   0, 0, 0, 1]])
    return Ft

def Wt_function(st, dt):
    Wt = np.array([[dt,                0],
                   [ 0, np.cos(st[0])*dt],
                   [ 0, np.sin(st[0])*dt],
                   [ 1,                0]])
    return Wt

# TODO: need to see how to implement landmark positions?
def Ht_function(st):
    Ht = zeros(6,4)
    st90 = st + np.array([-np.pi/2, 0, 0, 0])
    _, xy = takeMeasurement(st)
    xy_list.append(xy)
    _, xy = takeMeasurement(st90)
    xy_list.append(xy)

    for k in range(xy_list):
        dx = st[1]-x_list[k][0]
        dy = st[2]-y_list[k][1]
        d = np.sqrt(dx**2+dy**2)
        Ht[2*k:2*k+2,:] = np.array([[    dx/d,    dy/d,  0, 0],
                                    [-dy/d**2, dx/d**2, -1, 0]])

    Ht[5,:] = np.array([1, 0, 0, 0])
    Ht[6,:] = np.array([0, 0, 0, 1])

    return Ht



if __name__ == '__main__':
    import numpy as np

    st = np.array([np.pi/6, .400, .400])
    print(takeMeasurement(st))
    st = np.array([np.pi, .200, .400])
    print(takeMeasurement(st))
    st = np.array([np.pi*3/2, .200, .400])
    print(takeMeasurement(st))

dt = 0.001
t0 = 0
tf = 10
n_times = np.int(np.ceil((tf-t0)/dt))+1
u = np.ones((2, n_times))