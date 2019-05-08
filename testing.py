''' Sample Code for testing/debugging att module
'''

#imports
import numpy as np
from numpy import linalg as LA
import attitudes as att

quat1 = np.array([5,6,2,8])/LA.norm([5,6,2,8])

attitude1 = att.att._from_eul_ang(323,40,70,-10)._to_MRP()
attitude2 = att.att._from_PRV([1,2,3])._to_PRV()
attitude3 = att.att._from_quat([3,4,2,1]/LA.norm([3,4,2,1]))
attitude4 = att.att._from_quat(quat1)
attitude5 = att.att._from_CRP([quat1[1]/quat1[0],quat1[2]/quat1[0],quat1[3]/quat1[0]])
attitude6 = att.att._from_CRP([3,4,5])
attitude7 = att.att._from_MRP([quat1[1]/(1+quat1[0]),quat1[2]/(1+quat1[0]),quat1[3]/(1+quat1[0])])._to_MRP('long')
attitude8 = att.att._from_MRP([0.2,0.5,0.1])

hw1 = att.att._from_CRP([0.1,0.2,0.3])
hw2 = np.array([[0.333333,-0.666667,0.666667],
                [0.871795,0.487179,0.0512821],
                [-0.358974,0.564103,0.74359]])
hw2 = att.att(hw2)._to_CRP()

check1 = att.att._from_quat(quat1)._to_CRP()
check2 = att.att._from_quat(-quat1)._to_CRP()

hw3 = att.att._from_CRP([-0.1,-0.2,-0.3])
hw4 = att.att._from_CRP([-0.3,0.3,0.1])

def omega(t):
    return np.array([np.sin(0.1*t),0.01,np.cos(0.1*t)])*np.radians(3)

hw5 = att.att._from_CRP([0.4,0.2,-0.1])

soln = att.kinematics.integrator(omega,hw5,0,42)