name = "Attitudes"

''' Attitude Control Module

Ok, so the principle behind this module is the base class "att", which represents an attitude
description, by default of type "DCM". Can also be created from Euler Angles, PRV, Quaternions,
CRPs, and MRPs. Can also be transformed into these others by a method as well.

For simplicity's sake, I'm going to treat these classes as kind of a "dual number" where the DCM
representation is stored, but for all other types, the representation of that type is also stored.
 This should allow for direct quaternion addition and so forth.

 This should also allow me to simplify the addition/subtraction functions into a single function,
 that read the types of the inputs and acts accordingly.

 There will probably also be an angular acceleration vector class, but I'll get there when I get
 there.

 Author: Connor Johnstone
'''

#standard imports
import numpy as np
from numpy import linalg as LA


# ------------------------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------------------------
# Nothing here yet

# -----------------------------------------------------------------------------------------------
# BASE CLASS "ATT"
# -----------------------------------------------------------------------------------------------

class att():
    ''' Attitude Description Class
    Defines an attitude, by default from a DCM description. Also contains a whole bunch of class
    methods for defining by other means (CRP, quaternions, etc).

    Arguments:
        DCM: (ndarray [2x2]) General 3x3 DCM of the attitude description
    '''

    def __init__(self, DCM, type='DCM', angle_vec=np.array([]),units='rad',
                 euler_type=None,phi=None,path=None):
        ''' Standard Definition from a DCM '''
        if np.max(np.abs((DCM.T @ DCM) - np.eye(3))) > 1e-3:
            raise ValueError('DCM doesn\'t appear to be orthonormal')
        self.DCM = DCM
        self.type = type
        self.units = units
        if euler_type:
            self.order = euler_type
        if angle_vec != np.array([]):
            self.vec = angle_vec
        if phi:
            self.phi = phi
        if path:
            self.path = path

    def __repr__(self):
        if self.type == 'DCM':
            return 'DCM Attitude description is \n {}'.format(self.DCM)
        elif self.type == 'PRV':
            statement =  ''' \n
            {} Attitude description is: \n e = {} \n Phi = {} {} \n
            \n DCM description is: \n {} \n
            '''
            return statement.format(self.type,list(self.vec),self.phi,self.units,self.DCM)
        elif self.type == 'Euler Angle':
            statement =  '\n {} {} Attitude description is: \n {} {} \n \n DCM description is: \n {} \n'
            return statement.format(self.order,self.type,list(self.vec),self.units,self.DCM)
        else:
            statement =  '\n {} Attitude description is: \n {} \n \n DCM description is: \n {} \n'
            return statement.format(self.type,np.array(self.vec).flatten(),self.DCM)

    @classmethod
    def _from_eul_ang(cls,type,ang1,ang2,ang3,units='deg'):
        ''' Definition from Euler Angles

        Takes a type, 3 angles, and units to determine a DCM, then records both sets

        Arguments:
            type: (int) int of order of rotation axes
            ang1: (float) angle of rotation about first axis
            ang2: (float) angle of rotation about second axis
            ang3: (float) angle of rotation about third axis
            units: (string) either 'rad' or 'deg'
        '''
        if units=='deg':
            ang1, ang2, ang3 = np.radians(ang1),np.radians(ang2),np.radians(ang3)
        if type not in (123,132,213,231,312,321,131,121,212,232,313,323):
            raise ValueError('Euler angle type definition is incorrect')
        angle_vec = np.array([ang1,ang2,ang3])
        type = str(type)
        DCM = eul_to_DCM(int(type[0]),ang1,int(type[1]),ang2,int(type[2]),ang3,'rad')
        if units=='deg':
            angle_vec = np.degrees(angle_vec)
        return cls(DCM,'Euler Angle',angle_vec=angle_vec,units=units,euler_type=type)

    @classmethod
    def _from_PRV(cls,vec,phi=None,units='rad'):
        ''' Definition from Principle Rotation Vector

        Takes either a vector with norm != 1 or a normalized vector and a phi rotation magnitude
        Internally, the normalized vector and the phi rotation are used

        Arguments:
            vec: (list) principle rotation vector
            phi: (float) optional, rotation magnitude
            units: (string) either 'rad' or 'deg' to specify units for phi
        '''
        if not phi:
            phi = LA.norm(vec)
            vec = vec/LA.norm(vec)
        if units=='deg':
            phi = np.radians(phi)
        e1,e2,e3 = vec
        sigma = 1 - np.cos(phi)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        C = np.array([[e1*e1*sigma+cphi,e1*e2*sigma+e3*sphi,e1*e3*sigma - e2*sphi],
                      [e2*e1*sigma - e3*sphi,e2**2*sigma+cphi,e2*e3*sigma+e1*sphi],
                      [e3*e1*sigma+e2*sphi,e3*e2*sigma-e1*sphi,e3**2*sigma+cphi]])
        if units=='deg':
            phi = np.degrees(phi)
        return cls(C,'PRV', units=units, angle_vec=np.array(vec), phi=phi)

    @classmethod
    def _from_quat(cls,vec):
        '''Definition from Quaternions

        Takes in a quaternion and spits out an attitude object (DCM). Checks first for a valid
        quaternion

        Arguments:
            vec: (list) of quaternion values
        '''
        if np.abs(LA.norm(vec)-1) > 1e-13:
            raise ValueError('Quaternions must have norm of 1')
        b0,b1,b2,b3 = vec
        C = np.array([[b0**2+b1**2-b2**2-b3**2, 2*(b1*b2+b0*b3),            2*(b1*b3-b0*b2)],
                      [2*(b1*b2-b0*b3),         b0**2-b1**2+b2**2-b3**2,    2*(b2*b3+b0*b1)],
                      [2*(b1*b3+b0*b2),         2*(b2*b3-b0*b1),            b0**2-b1**2-b2**2+b3**2]])
        return cls(C,'Quaternion', angle_vec=vec)

    @classmethod
    def _from_CRP(cls,vec):
        '''Definition from Classical Rodriguez Parameters

        Uses the vector definition of the DCM to convert CRPs into a valid attitude object (element
        option also available in comments)

        Arguments:
            vec: (list) of CRP values
        '''
        q = np.atleast_2d(vec).reshape(3,1)
        C = (1/(1+q.T@q))*((1-q.T@q)*np.eye(3) + 2 * q @ q.T - 2 * tilde(q))
#        q1,q2,q3 = q.reshape(np.size(vec))
#        C = np.array([[1+q1**2-q2**2-q3**2, 2*(q1*q2+q3),           2*(q1*q3-q2)],
#                      [2*(q1*q2-q3),        1-q1**2+q2**2-q3**2,    2*(q2*q3+q1)],
#                      [2*(q1*q3+q2),        2*(q2*q3-q1),           1-q1**2-q2**2+q3**2]])
#        C = (1/(1 + q.T @ q)) * C
        return cls(C,'CRP',angle_vec=np.array(vec))

    @classmethod
    def _from_MRP(cls,vec):
        '''Definition from Modified Rodriguez Parameters

        Uses the vector definition of the DCM to convert MRPs into a valid attitude object. Returns
        the path whether it's long (norm > 1) or short (norm < 1) with norm==1 taken to be short

        Arguments:
            vec: (list) of MRP values

        '''
        s = np.atleast_2d(vec).T
        C = np.eye(3) + (8*tilde(s)@tilde(s) - 4*(1-s.T@s)*tilde(s))/(1+s.T@s)**2
        if LA.norm(vec) > 1:
            path = 'long'
        else:
            path = 'short'
        return cls(C,'MRP',angle_vec=np.array(vec),path=path)

    def _to_eul_ang(self,type,units='deg'):
        '''Conversion to Euler Angles. There's no easy way to do this, so it's always just done
        from the DCM. Which is fine, it's still quick.

        Arguments:
            type: (int) currently must be 321 or 313 since those are common. Will expand
            units: (str) optional, units to output the angles
        '''
        C = self.DCM
        if type == 321:
            ang1 = np.arctan2(C[0,1],C[0,0])
            ang2 = -np.arcsin(C[0,2])
            ang3 = np.arctan2(C[1,2],C[2,2])
        elif type == 313:
            ang1 = np.arctan2(C[2,0],-C[2,1])
            ang2 = np.arccos(C[2,2])
            ang3 = np.arctan2(C[0,2],C[1,2])
        if units == 'deg':
            ang1,ang2,ang3 = np.degrees([ang1,ang2,ang3])
        return self._from_eul_ang(type,ang1,ang2,ang3,units=units)

    def _to_PRV(self, units='rad'):
        '''Conversion to Principle Rotation Vector. Always done from the DCM. Doesn't need to
        take any arguments

        Outputs the short version of the PRV (using arccos function) and the positive output
        for e_hat
        '''
        C = self.DCM
        phi = np.arccos(0.5*(C[0,0]+C[1,1]+C[2,2]-1))
        e = (1/(2*np.sin(phi)))*np.array([C[1,2]-C[2,1],C[2,0]-C[0,2],C[0,1]-C[1,0]])
        if units=='deg':
            phi = np.degrees(phi)
        return self._from_PRV(e,phi=phi,units=units)

    def _to_quat(self, path='short'):
        '''If the object is a classical or modified Rodriguez parameter object, directly converts
        to quaternions via known relations. Otherwise, uses sheppard's method to determine the
        quaternions from the DCM.

        Arguments:
            path: (str) optional, tells the function whether you'd like the short way or the
            long way
        '''
        if self.type == 'CRP':
            q = self.vec
            b0 = 1/np.sqrt(1+LA.norm(q)**2)
            b1 = q[0]*b0
            b2 = q[1]*b0
            b3 = q[2]*b0
        elif self.type == 'MRP':
            s = self.vec
            b0 = (1-LA.norm(s)**2)/(1+LA.norm(s)**2)
            b1 = 2*s[0]/(1+LA.norm(s)**2)
            b2 = 2*s[1]/(1+LA.norm(s)**2)
            b3 = 2*s[2]/(1+LA.norm(s)**2)
        else:
            #the annoying way...
            C = self.DCM
            [[C11,C12,C13],
             [C21,C22,C23],
             [C31,C32,C33]] = C
            trC = C[0,0]+C[1,1]+C[2,2]
            b02 = 0.25*(1+trC)
            b12 = 0.25*(1+2*C[0,0]-trC)
            b22 = 0.25*(1+2*C[1,1]-trC)
            b32 = 0.25*(1+2*C[2,2]-trC)
            b0b1 = (C23 - C32)/4
            b0b2 = (C31 - C13)/4
            b0b3 = (C12 - C21)/4
            b1b2 = (C12 + C21)/4
            b3b1 = (C31 + C13)/4
            b2b3 = (C23 + C32)/4
            squares = [b02,b12,b22,b32]
            if b02 == np.max(squares):
                b0 = np.sqrt(b02)
                b1 = b0b1/b0
                b2 = b0b2/b0
                b3 = b0b3/b0
            elif b12 == np.max(squares):
                b1 = np.sqrt(b12)
                b0 = b0b1/b1
                b2 = b1b2/b1
                b3 = b3b1/b1
            elif b22 == np.max(squares):
                b2 = np.sqrt(b22)
                b0 = b0b2/b2
                b1 = b1b2/b2
                b3 = b2b3/b2
            else:
                b3 = np.sqrt(b32)
                b0 = b0b3/b3
                b1 = b3b1/b3
                b2 = b2b3/b3
        quats = np.array([b0,b1,b2,b3])
        if b0 > 0 and path == 'long':
            quats = -quats
        elif b0 < 0 and path == 'short':
            quats = -quats
        return self._from_quat(quats)

    def _to_CRP(self):
        '''Conversion to Classical Rodriguex Parameters. If the initial attitude is in quaternions,
         then it converts directly, because that's very easy. Otherwise, it converts from the DCM,
         which is actually still pretty easy. No arguments because the shadow set doesn't really
         exist.
         '''
        if self.type == 'Quaternion':
            b0,b1,b2,b3 = self.vec
            q = np.array([b1/b0,b2/b0,b3/b0])
        else:
            C = self.DCM
            [[C11,C12,C13],
             [C21,C22,C23],
             [C31,C32,C33]] = C
            zeta = np.sqrt(C11+C22+C33+1)
            q = (1/zeta**2)*np.array([C23-C32,C31-C13,C12-C21])
        return self._from_CRP(q)

    def _to_MRP(self,path='short'):
        '''Conversion to Modified Rodriguez Parameters

        Similar to CRPs, if the input attitude is a quaternion, it'll just do the output directly,
         otherwise, it'll compute the CRP from the DCM. This function does have an input for the
        short rotation or the long rotation, though.
        '''
        if self.type == 'Quaternion':
            b0,b1,b2,b3 = self.vec
            s = np.array([b1/(1+b0),b2/(1+b0),b3/(1+b0)])
        else:
            C = self.DCM
            [[C11,C12,C13],
             [C21,C22,C23],
             [C31,C32,C33]] = C
            zeta = np.sqrt(C11+C22+C33+1)
            s = (1/(zeta*(zeta+2)))*np.array([C23-C32,C31-C13,C12-C21])
        if LA.norm(s) > 1 and path=='short':
            s = -s/LA.norm(s)
        elif LA.norm(s) < 1 and path=='long':
            s = -s/LA.norm(s)
        return self._from_MRP(s)


# ------------------------------------------------------------------------------------------------
# INTERNAL FUNCTIONS (designed to be used by module, not user)
# ------------------------------------------------------------------------------------------------
def rot(angle,axis,radordeg):
    ''' Defines a single axis rotation'''
    mat = np.array([])

    if radordeg == 'rad':
        angle = angle
    elif radordeg == 'deg':
        angle = np.radians(angle)
    else:
        print('Error')

    if axis==1:
        mat = np.array( [[  1,    0,                 0                ],
                         [  0,    np.cos(angle),     np.sin(angle)   ],
                         [  0,    -np.sin(angle),     np.cos(angle)    ]])
    elif axis==2:
        mat = np.array( [[  np.cos(angle),   0,  -np.sin(angle)  ],
                         [  0,               1,  0              ],
                         [  np.sin(angle),  0,  np.cos(angle)  ]])
    elif axis==3:
        mat = np.array([[   np.cos(angle),   np.sin(angle),    0   ],
                        [   -np.sin(angle),   np.cos(angle),     0   ],
                        [   0,               0,                 1   ]])
    else:
        print('Error')

    return mat

def eul_to_DCM(rot1axis,rot1ang,rot2axis,rot2ang,rot3axis,rot3ang,radordeg):
    '''Combines 3 axis rotations to complete a DCM from euler angles'''
    mat1 = rot(rot1ang,rot1axis,radordeg)
    mat2 = rot(rot2ang,rot2axis,radordeg)
    mat3 = rot(rot3ang,rot3axis,radordeg)
    DCM = mat3@mat2@mat1
    return DCM.T

# ------------------------------------------------------------------------------------------------
# MAIN FUNCTIONS (to be used by module or user)
# ------------------------------------------------------------------------------------------------

def tilde(x):
    '''
    Returns a tilde matrix for a given vector. Should be robust enough to handle vectors of
    any reasonable (vector-like) shape
    '''
    x = np.array(x).reshape(np.size(x))
    return np.array([[0,        -x[2],      x[1]    ],
                     [x[2],     0,          -x[0]   ],
                     [-x[1],    x[0],       0       ]])

def add(att1,att2):
    ''' Addition function between attitude descriptions.

    This function will first check to see whether the addition can be done directly (generally
    when the two parameters are off the same type) and will do it that way if so. However, I am
    skipping symmetric Euler Angle addition for now. If direct addition cannot be done, then the
    DCM is used to add the two parameters and the output type can be chosen by the user.

    ONLY RETURNS DCM OUTPUT UNLESS DIRECT ADDITION IS ALLOWED (WILL BE CHANGED IN FUTURE)

    Arguments:
        att1: (att object) representing the first attitude to sum
        att2: (att object) representing the second attitude to sum
    '''
    if att1.type=='PRV' and att2.type=='PRV':
        phi1 = att1.phi
        phi2 = att2.phi
        e1 = att1.vec
        e2 = att2.vec
        phi = 2*np.arccos(np.cos(phi1/2)*np.cos(phi2/2)-np.sin(phi1/2)*np.sin(phi2/2)*np.dot(e1,e2))
        e = (np.cos(phi2/2)*np.sin(phi1/2)*e1+np.cos(phi1/2)*np.sin(phi2/2)*e2+\
             np.sin(phi1/2)*np.sin(phi2/2)*np.cross(e1,e2))/np.sin(phi/2)
        return att._from_PRV(e,phi)
    elif att1.type=='Quaternion' and att2.type=='Quaternion':
        b0_1,b1_1,b2_1,b3_1 = att1.vec
        b0_2,b1_2,b2_2,b3_2 = att2.vec
        b_1 = np.array(att1.vec).reshape(4,1)
        mat = np.array([[b0_2,  -b1_2,  -b2_2,  -b3_2    ],
                        [b1_2,  b0_2,   b3_2,   -b2_2    ],
                        [b2_2,  -b3_2,  b0_2,   b1_2     ],
                        [b3_2,  b2_2,   -b1_2,  b0_2     ]])
        return att._from_quat((mat @ b_1).reshape(4))
    elif att1.type=='CRP' and att2.type=='CRP':
        q1 = att1.vec
        q2 = att2.vec
        q = (q2 + q1 - np.cross(q2,q1))/(1-np.dot(q2,q1))
        return att._from_CRP(q)
    elif att1.type=='MRP' and att2.type=='MRP':
        #outputs the "short" MRP
        s2 = att1.vec
        s1 = att2.vec
        s = ((1-LA.norm(s1)**2)*s2 + (1-LA.norm(s2)**2)*s1 - 2*np.cross(s2,s1))/(1 + \
            LA.norm(s1)**2*LA.norm(s2)**2 - 2*np.dot(s1,s2))
        return att._from_MRP(s)
    else:
        C1 = att1.DCM
        C2 = att2.DCM
        C = C2 @ C1
        return att(C,'DCM')

def subtract(att1,att2):
    ''' Subtraction function between attitude descriptions.

    This function will first check to see whether the subtraction can be done directly (generally
    when the two parameters are off the same type) and will do it that way if so. However, I am
    skipping symmetric Euler Angle subtraction for now. If direct addition cannot be done, then the
    DCM is used to subtract the two parameters and the output type can be chosen by the user.

    This function subtracts att1 from att2

    ONLY RETURNS DCM OUTPUT UNLESS DIRECT SUBTRACTION IS ALLOWED (WILL BE CHANGED IN FUTURE)

    Arguments:
        att1: (att object) representing the first attitude to difference
        att2: (att object) representing the second attitude to difference
    '''
    if att1.type=='PRV' and att2.type=='PRV':
        phi1 = att1.phi
        phi2 = att2.phi
        e1 = att1.vec
        e2 = att2.vec
        phi = 2*np.arccos(np.cos(phi1/2)*np.cos(phi2/2)+np.sin(phi1/2)*np.sin(phi2/2)*np.dot(e2,e1))
        e = (np.cos(phi1/2)*np.sin(phi2/2)*e2-np.cos(phi2/2)*np.sin(phi1/2)*e1+\
             np.sin(phi1/2)*np.sin(phi2/2)*np.cross(e2,e1))/np.sin(phi/2)
        return att._from_PRV(e,phi)
    elif att1.type=='Quaternion' and att2.type=='Quaternion':
        b0_1,b1_1,b2_1,b3_1 = att1.vec
        b0_2,b1_2,b2_2,b3_2 = att2.vec
        b_2 = np.array(att2.vec).reshape(4,1)
        mat = np.array([[b0_1,  -b1_1,  -b2_1,  -b3_1   ],
                        [b1_1,  b0_1,   -b3_1,  b2_1    ],
                        [b2_1,  b3_1,   b0_1,   -b1_1   ],
                        [b3_1,  -b2_1,  b1_1,   b0_1    ]]).T
        return att._from_quat((mat @ b_2).reshape(4))
    elif att1.type=='CRP' and att2.type=='CRP':
        q1 = att1.vec
        q2 = att2.vec
        q = (q2 - q1 + np.cross(q2,q1))/(1+np.dot(q2,q1))
        return att._from_CRP(q)
    elif att1.type=='MRP' and att2.type=='MRP':
        #outputs the "short" MRP
        s1 = att1.vec
        s2 = att2.vec
        s = ((1-LA.norm(s1)**2)*s2 - (1-LA.norm(s2)**2)*s1 + 2*np.cross(s2,s1))/(1 + \
            LA.norm(s1)**2*LA.norm(s2)**2 + 2*np.dot(s1,s2))
        return att._from_MRP(s)
    else:
        C1 = att1.DCM
        C2 = att2.DCM
        C = C2 @ C1.T
        return att(C,'DCM')

def invert(att_desc):
    ''' Inverts an attitude, depending on its type. Does it directly if it can. Otherwise, via the
    DCM

    Arguments:
        att_desc: (att object) the attitude to invert
    '''
    if att_desc.type == 'PRV':
        phi = -att_desc.phi
        return att._from_PRV(att_desc.vec, phi, units = att_desc.units)
    elif att_desc.type == 'Quaternions':
        b0,b1,b2,b3 = att_desc.vec
        return att._from_quat(np.array([b0,-b1,-b2,-b3]))
    elif att_desc.type == 'CRP':
        return att._from_CRP(-np.array(att_desc.vec))
    elif att_desc.type == 'MRP':
        return att._from_MRP(-np.array(att_desc.vec))
    elif att_desc.type == 'Euler Angle':
        return att(att_desc.DCM.T)._to_eul_ang(att_desc.order,units=att_desc.units)
    elif att_desc.type == 'DCM':
        return att(att_desc.DCM.T)
    else:
        raise ValueError('Not a valid attitude  description')

def shadow(att_desc):
    ''' Computes the "shadow set" for a given Quaternion or MRP ONLY

    Arguments:
        att_desc: (att object) the attitude in question
    '''
    if att_desc.type == 'Quaternion':
        return att._from_quat(-np.array(att_desc.vec))
    elif att_desc.type == 'MRP':
        s = np.array(att_desc.vec)
        new_s = -s/LA.norm(s)**2
        return att._from_MRP(new_s)
