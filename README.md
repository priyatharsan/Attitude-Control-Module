# Attitude-Control-Module
This attitude module is written to be a broadly useful way to perform simple
attitude determination/kinematics/controls tasks in a python environment. For
the most part, the class centers around the "att" class, which can take the
form:

- DCM
- Euler Angles (3-1-3 and 3-2-1 well supported)
- Principle Rotation Vectors
- Quaternions
- Classical Rodriguez Parameters
- Modified Rodriguez Parameters

with the idea being that this module allows for easy translation from one type
of attitude description to another, and includes a host of functions and methods
that can apply to any of the types more or less equally.

# Basic Arithmetic Functions
The module supports:

- creation of a tilde (cross) matrix of a vector
- addition of any two attitude descriptions (of any type with no transformation
  necessary)
- subtraction of any two attitude descriptions (of any type with no transformation
  necessary)
- attitude description "inversion"

# Attitude Determination Functions
This module supports several different methods of determining a reference frame
using two attitude measurements. These are:

- TRIAD Method
- Devenport's Q-Method
- QUEST Method
- OLAE (Batch LLS) Method

All of these are supported for each of the attitude types

# Kinematics Functions
This module supports the differential kinematic equations for each of the
supported attitude types in both rigid body and general body form. It also
includes a built-in integrator that supports RK4 integration special
considerations for attitude types such as Modified Rodriguez Parameter "shadow
sets"
