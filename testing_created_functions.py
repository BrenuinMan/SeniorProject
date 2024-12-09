'''Testing file for created functions in other files.'''
import grcwa
import numpy as np
import time
import matplotlib.pyplot as plt

from example.uniform_sphere import uniformsphere

'''Input parameters here'''
nG = 101                    # Truncation order
L1 = [0.8,0]                # Lattice constant (x direction)
L2 = [0,0.8]                # Lattice constant (y direction)
theta = 0                   # Incidence light angle

# Patterned layer gridding (Nx*Ny)
Nx = 100
Ny = 100

Np = 99                     # Number of patterned layers
epbkg = 4                  # Dielectric value of uniform sphere
diameter = 1.0              # diameter of sphere

R,T = uniformsphere(nG,L1,L2,theta,Nx,Ny,Np,epbkg,diameter)
print('R=',R,', T=',T,', R+T=',R+T)

'''This section is for graphing, not always needed, takes about 10 seconds per calculation'''
'''
Rarray = []
Tarray = []

for index in range(0,len(theta)):
    R,T = uniformsphere(nG,L1,L2,theta[index],Nx,Ny,Np,epbkg,diameter)
    Rarray.append(R)
    Tarray.append(T)
    print('R=',R,', T=',T,', R+T=',R+T)
print("Complete")

# Creating plot
plt.plot(np.sin(theta), Rarray)
plt.plot(np.sin(theta), Tarray, '-.')

plt.xlabel("Theta")
plt.ylabel("R and T")
plt.legend(["R","T"])
plt.title('Change in Incidence Angle of Light Resulting in Reflection and Transmission')
plt.show()
'''
