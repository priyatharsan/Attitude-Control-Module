''' Attitude Determination Module

Most of these functions allow for a couple or more of input attitudes and will spit out a DCM
attitude object that relates the body frame to the inertial frame.

This module can perform attitude determination using:
    
     - TRIAD Method
     - Devenport's Q Method
     - QUEST Method
     - OLEA Method
'''

#imports
import numpy as np
from scipy import linalg as LA
from scipy.optimize import newton
from attitudes import tilde, subtract
from attitudes import att as Att

# ------------------------------------------------------------------------------------------------
# ATTITUDE DETERMINATION FUNCTIONS
# ------------------------------------------------------------------------------------------------

def triad(meas1_b, meas2_b, meas1_n, meas2_n):
    '''TRIAD method. Basically define a reference frame using two of your measurements (and their
    known positions in space) and use that to get your body frame.
    
    Arguments:
        meas1_b: (ndarray) of measurement 1 (the better measurement) in the b-frame
        meas2_b: (ndarray) of measurement 2 (the worse measurement) in the b-frame
        meas1_n: (ndarray) of measurement 1 (the better measurement) in the inertial-frame
        meas2_n: (ndarray) of measurement 2 (the worse measurement) in the inertial-frame
    '''
    t1_b = np.array(meas1_b)/LA.norm(meas1_b)
    t2_b = np.cross(t1_b,meas2_b)/LA.norm(np.cross(t1_b,meas2_b))
    t3_b = np.cross(t1_b,t2_b)
    t1_n = np.array(meas1_n)/LA.norm(meas1_n)
    t2_n = np.cross(t1_n,meas2_n)/LA.norm(np.cross(t1_n,meas2_n))
    t3_n = np.cross(t1_n,t2_n)
    
    t1_b = t1_b.reshape(3,1)
    t2_b = t2_b.reshape(3,1)
    t3_b = t3_b.reshape(3,1)
    t1_n = t1_n.reshape(3,1)
    t2_n = t2_n.reshape(3,1)
    t3_n = t3_n.reshape(3,1)
    
    bt = np.append(np.append(t1_b,t2_b,axis=1),t3_b,axis=1)
    BT = Att(bt)
    nt = np.append(np.append(t1_n,t2_n,axis=1),t3_n,axis=1)
    NT = Att(nt)
    
    return subtract(NT,BT)

def q_method(meas_b, meas_n, weights=None):
    ''' Devenport's Q-Method
    Determine the solution to Wahba's problem by taking the eigenvalues and eigenvectors of the
    K matrix. May be slower than QUEST.
    
    Arguments:
        meas_b: (list) of ndarrays of b-frame measurements
        meas_n: (list) of ndarrays of inertial-frame measurements
        weights: (list) optional list of weights
    '''
    if not weights:
        weights = np.ones(len(meas_b))
    B = np.zeros((3,3))
    for i in range(len(meas_b)):
        meas_b[i] = meas_b[i].reshape(3,1)/LA.norm(meas_b[i])
        meas_n[i] = meas_n[i].reshape(3,1)/LA.norm(meas_n[i])
        B = B + weights[i]*meas_b[i] @ meas_n[i].T
    Z = np.array([[B[1,2]-B[2,1],B[2,0]-B[0,2],B[0,1]-B[1,0]]]).T
    s = B[0,0]+B[1,1]+B[2,2]
    S = B + B.T
    K = np.append(np.append([[s]],Z.T,axis=1),np.append(Z,S-s*np.eye(3),axis=1),axis=0)
    vals, vecs = LA.eig(K)
    g = vecs[np.argmax(vals),:]
    if g[0] < 0:
        g = -g
    return Att._from_quat(g)

def quest(meas_b, meas_n, weights=None):
    ''' QUEST Method
    
    Exactly the same as Devenport's Q-Method, except that it solves the eigenvalue problem 
    numberically, using the sum of the weights as a very good initial estimate.
    
    Arguments:
        meas_b: (list) of ndarrays of b-frame measurements
        meas_n: (list) of ndarrays of inertial-frame measurements
        weights: (list) optional list of weights
    '''
    if not weights:
        weights = np.ones(len(meas_b))
    B = np.zeros((3,3))
    for i in range(len(meas_b)):
        meas_b[i] = meas_b[i].reshape(3,1)/LA.norm(meas_b[i])
        meas_n[i] = meas_n[i].reshape(3,1)/LA.norm(meas_n[i])
        B = B + weights[i]*meas_b[i] @ meas_n[i].T
    Z = np.array([[B[1,2]-B[2,1],B[2,0]-B[0,2],B[0,1]-B[1,0]]]).T
    s = B[0,0]+B[1,1]+B[2,2]
    S = B + B.T
    K = np.append(np.append([[s]],Z.T,axis=1),np.append(Z,S-s*np.eye(3),axis=1),axis=0)
    def f(s):
        return LA.det(K-s*np.eye(4))
    lam = newton(f,np.sum(weights))
    q = LA.inv((lam+s)*np.eye(3) - S) @ Z
    return Att._from_CRP(q)

def olae(meas_b, meas_n, weights=None):
    ''' OLAE Method: literally just a batch LLS solution of a reformulated Cayley Transform eqn.
    Linear equation, no Wahba's cost function. So very fast.
    
    Arguments:
        meas_b: (list) of ndarrays of b-frame measurements
        meas_n: (list) of ndarrays of inertial-frame measurements
        weights: (list) optional list of weights
    '''
    if not weights:
        weights = np.ones(len(meas_b))
    meas_b[0] = meas_b[0].reshape(3,1)/LA.norm(meas_b[0])
    meas_n[0] = meas_n[0].reshape(3,1)/LA.norm(meas_n[0])
    d = np.array(meas_b[0]-meas_n[0])
    S = tilde(meas_b[0]+meas_n[0])
    W = weights[0]*np.eye(3)
    for i in range(1,len(meas_b)):
        meas_b[i] = meas_b[i].reshape(3,1)/LA.norm(meas_b[i])
        meas_n[i] = meas_n[i].reshape(3,1)/LA.norm(meas_n[i])
        d = np.append(d,meas_b[i]-meas_n[i],axis=0)
        S = np.append(S,tilde(meas_b[i]+meas_n[i]),axis=0)
        W = LA.block_diag(W,weights[i]*np.eye(3))
    q = LA.inv(S.T @ W @ S) @ S.T @ W @ d
    return Att._from_CRP(q)
