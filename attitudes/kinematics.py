''' Attitude Control Kinematic Equations

This should actually be a single equation that can handle any attitude description. Basically,
you pass it a function for the angular velocity vector, and it spits out the derivative of the
attitude description. This will be implemented in reverse in the future, as well.

I'm also going to include a special Runge-Kutta 4th Order integrator that is helpful for 
integrating CRP states (since they have to be handled a little differently than normal vectors
or whatever).

'''

#Ok, here we go...

#imports
import numpy as np
from numpy import linalg as LA
from attitudes import tilde

# ------------------------------------------------------------------------------------------------
# THE ONE KINEMATIC EQUATION TO RULE THEM ALL
# ------------------------------------------------------------------------------------------------

def kin_eq(att,t,omega):
    ''' Kinematic Equation for a generic attitude description and Ang Vel function.
    
    Basically, just pass an attitude, a time, and a function for angular velocity and you get back
    a time derivative of the attitude. You definitely want to pass this to the built-in Runge-Kutta
    integrator if you're going to be integrating numerically, since it handles some special cases 
    in a way that a normal integrator won't.
    
    DCM attitude not yet supported
    
    Arguments:
        att: (att object) attitude description at time t
        t: (int, float) time
        omega: (function) that should return a size(3) ndarray vector of the angular velocity
    '''
    w = np.array(omega(t)).reshape(3,1)
    if att.type == 'Euler Angle':
        if att.units == 'deg':
            vec = np.radians(att.vec)
        else:
            vec = att.vec
        ang1, ang2, ang3 = vec
        if att.order == 321:
            B = np.array([[0,               np.sin(ang3),               np.cos(ang3)              ],
                          [0,               np.cos(ang3)*np.cos(ang2),  -np.sin(ang3)*np.cos(ang2)],
                          [np.cos(ang2),    np.sin(ang3)*np.sin(ang2),  np.cos(ang3)*np.sin(ang2) ]])*(1/np.cos(ang2))
        elif att.order == 313:
            B = np.array([[np.sin(ang3),               np.cos(ang3),               0            ],
                          [np.cos(ang3)*np.sin(ang2),  -np.sin(ang3)*np.sin(ang2), 0            ],
                          [-np.sin(ang3)*np.cos(ang2), -np.cos(ang3)*np.cos(ang2), np.sin(ang2) ]])*(1/np.sin(ang2))
    elif att.type == 'PRV':
        phi = att.phi
        if att.units == 'deg':
            phi = np.radians(phi)
        g = np.array(att.vec)*phi
        B = (np.eye(3) + 0.5*tilde(g) + (1/phi**2)*(1-(phi/2)*(1/np.tan(phi/2)))*tilde(g) @ tilde(g))
    elif att.type == 'Quaternion':
        b0,b1,b2,b3 = att.vec
        B = 0.5*np.array([[ -b1, -b2,    -b3   ],
                          [ b0,  -b3,    b2    ],
                          [ b3,  b0,     -b1   ],
                          [ -b2, b1,     b0    ]])
    elif att.type == 'CRP':
        q = np.array(att.vec).reshape(3,1)
        B = 0.5 * (np.eye(3) + tilde(q) + q @ q.T)
    elif att.type == 'MRP':
        s = np.array(att.vec).reshape(3,1)
        B = 0.25*((1-LA.norm(s)**2)*np.eye(3) + 2*tilde(s) + 2*s @ s.T)
    else:
        raise ValueError('Not a valid attitude type (DCM not supported yet)')
    return B @ w

# ------------------------------------------------------------------------------------------------
# INTEGRATOR FUNCTION
# ------------------------------------------------------------------------------------------------

def integrator(omega,init_att,init_time,final_time,time_step=0.1):
    ''' This is an integrator designed explicitly to work with attitude objects. It can handle some
    special  cases, like CRP shadow-switching, and always uses the kin_eq function (hence no
    kinematic equation function argument). 
    
    Arguments:
        omega: (function) that should return a size(3) ndarray vector of the angular velocity at time t
        init_att: (att) initial attitude
        init_time: (float) initial time
        final_time: (float) final time
        time_step: (float) integrator time step
    '''
    
    output = np.append(init_att.vec.reshape(1,np.size(init_att.vec)),[[init_time]], axis=1)
    time = init_time
    att = init_att
    
    while time <= final_time:
        current_vec = att.vec
        k1 = (kin_eq(att,time,omega)*time_step)
        k1 = k1.reshape(np.size(k1))
        att.vec = att.vec + k1/2
        k2 = kin_eq(att,time+time_step/2,omega)*time_step
        k2 = k2.reshape(np.size(k2))
        att.vec = att.vec+k2/2
        k3 = kin_eq(att,time+time_step/2,omega)*time_step
        k3 = k3.reshape(np.size(k3))
        att.vec = att.vec+k3
        k4 = kin_eq(att,time+time_step,omega)*time_step
        k4 = k4.reshape(np.size(k4))
        
        att.vec = current_vec + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        if att.type == 'MRP' and LA.norm(att.vec) > 1:
            att.vec = -att.vec/LA.norm(att.vec)**2
        
        time += time_step
        next_row = np.append(init_att.vec.reshape(1,np.size(init_att.vec)),[[time]], axis=1)
        output = np.append(output,next_row,axis=0)
        
    return output