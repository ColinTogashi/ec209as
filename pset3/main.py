import numpy as np
from common import state


wheel_radius = 0.02 # [m]
wheel_separation_distance = 0.085 # [m]

L = 0.75 # [m]
W = 0.5 # [m]

def convertDecoupledInputsToCoupledInputs(ut):

    # Note: defined first input is left motor, second is right
    wt = (ut[1]-ut[0])/2.0
    vt = (ut[0]+ut[1])/2.0
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
    def __init__(self,dt):
        pass


def timeUpdate(st,ut,dt,S,Q):
    wt, vt = convertDecoupledInputsToCoupledInputs(ut)
    Ft = np.array([[                   1, 0, 0],
                   [-vt*np.sin(st[0])*dt, 1, 0],
                   [ vt*np.cos(st[0])*dt, 0, 1]])
    Wt = np.array([[dt,                0],
                   [ 0, np.cos(st[0])*dt],
                   [ 0, np.sin(st[0])*dt]])

    st_plus_1 = Ft*st
    S_plus_1 = Ft*S*Ft.T + Wt*Q*Wt.T 
    return xt_plus_1, S_plus_1

def measurementUpdate(st,dt,S_inv,x_l,y_l,R):
    Ht = np.array([[    (st[1]-x_l)/dt,    (st[2]-y_l)/dt,  0],
                   [-(st[2]-y_l)/dt**2, (st[1]-x_l)/dt**2, -1]])
    S_inv = S_inv + Ht.T*np.linalg.inv(R)*H


if __name__ == '__main__':
    import numpy as np

    st = np.array([np.pi/6, .400, .400])
    print(takeMeasurement(st))
    st = np.array([np.pi, .200, .400])
    print(takeMeasurement(st))
    st = np.array([np.pi*3/2, .200, .400])
    print(takeMeasurement(st))