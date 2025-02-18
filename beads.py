from IndexDict import IndexDict
import numpy as np
import matplotlib.pyplot as plt 
import grcwa

"""Transmission and reflection of a uniform sphere."""
import time
import math

def honeycomb_lattice(obj,Nx,Ny,Np,eps,diameter):
    thickp = diameter/Np                                                # thickness of patterned layer

    radius_top = np.linspace(0,0.5,int((Np+1)/2))              # radius of the top half of the sphere in increments for layer creation
    radius_bot = np.linspace(0.5,0,int((Np+1)/2))              # radius of the bottom half of the sphere in increments for layer creation
    delta_radius = radius_top[1]                                        # the delta radius will be equal to the first value, which is a difference of said value and zero

    # eps for patterned layer
    x0 = np.linspace(0,math.sqrt(3),Nx)
    y0 = np.linspace(0,1.,Ny)
    x, y = np.meshgrid(x0,y0,indexing='ij')

    """This section will determine the limit to which the radius will extend to, and makes sure the middle section"""
    """won't be repeated if there are an odd number of layers"""
    # This will check if the number of layers is odd and make sure the middle layer isn't repeated
    radius_limit = 0                                                    # The radius_limit will initially be set to zero
    if int(Np%2) == 1:                                                  # Modular divide and if we have a remainder of 1, we have an odd number of layers
        radius_limit = radius_top[int((Np+1)/2)-1]                      # Our radius limit won't repeat the middle layer when it's odd
        index = len(radius_top)-2                                       # Index starts one before the final value for the bottom index
        reverseindex = 0
        while (radius_top[index]!=0):
            radius_bot[reverseindex] = radius_top[index]                # Reverses the radius_top values and puts them into the radius_bot
            index-=1
            reverseindex+=1
        radius_bot[reverseindex] = 0                                    # Sets the last value to zero because I can't get np.delete to work

    top = True                                                          # Helps flip our condition so we start looking at the bottom layer
    for i in range(Np-2):
        obj.Add_LayerGrid(thickp,Nx,Ny)
        print(f"Layer grid added")

    topindex = 1                                                        # We don't want to consider a radius of zero, so we start at the second value
    botindex = 0                                                        # Since our bottom index needs to know the first value, we start at zero
    epgrid = np.array([])                                               # We start with an empty array to append the flatten integer arrays into

    start_time = time.time()

    for layer in range(0,Np-2):                                         # This adds Np layers with half increasing to radius_limit and the rest decreasing to zero
        if top == True:                                                 # This section covers the top half of the sphere
            if radius_limit == 0:                                       # This covers when there's an even # of layers
                delta_radius = radius_top[topindex]                     # The radius we're considering changes through every iteration
                if topindex >= int((Np-1)/2):                           # This marks the halfway point that will move us to the bottom half
                    top = False
            else:                                                       # This covers when there's an odd # of layers
                delta_radius = radius_top[topindex]                     # The radius we're considering changes through every iteration
                if topindex >= int((Np)/2):                             # This marks the halfway point that will move us to the bottom half
                    top = False
            topindex += 1                                               # This moves us to analyze the next top layer

        else:                                                           # This section covers the bottom half of the sphere (bottom = True)
            if radius_limit == 0:                                       # This covers when there's an even # of layers
                delta_radius = radius_bot[botindex]                     # The radius we're considering changes through every iteration
                if botindex >= int((Np-1)/2):                           # Once this condition is true, we have finished the calculation
                    print("FINISHED EVEN")
                    break
            else:
                delta_radius = radius_bot[botindex]                     # The radius we're considering changes through every iteration
                if botindex >= int((Np-2)/2):                           # Once this condition is true, we have finished the calculation
                    print("FINISHED ODD")
                    break
            botindex += 1                                               # This moves us to analyze the next bottom layer
        'The next 4 lines create the layers that we are analyzing'
        epname = np.ones((Nx, Ny), dtype=complex)
        honeycomb = np.logical_or(((x-.866)**2 + (y-.5)**2 < delta_radius**2), (x**2 + y**2 < delta_radius**2))
        honeycomb = np.logical_or(honeycomb, ((x-1.73205)**2) + y**2 < delta_radius**2)
        honeycomb = np.logical_or(honeycomb, (x**2 + (y-1)**2) < delta_radius**2)
        honeycomb = np.logical_or(honeycomb, ((x-1.73205)**2 + (y-1)**2) < delta_radius**2)
        epname[honeycomb] = eps
        epgrid = np.append(epgrid.flatten(),epname.flatten())
        # from code import interact
        # interact(local=locals())
        
        # plt.contourf(x,y,honeycomb,cmap='jet')
        # plt.axis('equal')
        # plt.tight_layout
        # plt.show()
        
    return epgrid

    # We combine all of the epsilon values
    obj.GridLayer_geteps(epgrid)

    # We create the planewave excitation
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    # solve for R and T
    R,T= obj.RT_Solve(normalize=1)
    print("This program took %s seconds to run" % round((time.time() - start_time),4))

    return R, T

nm_to_um = 1e-3

dev_structure = [
    ('air',0.0*nm_to_um,'slab'),
    ('sio2',8,'honeycomb'),
    ('sio2',1000,'slab'),
    ('air',0.0*nm_to_um,'slab')
]
start_wv = 5  # 5 um
end_wv = 20   # 20 um
wv_sweep = np.linspace(start_wv, end_wv, num=100, endpoint=True)

IndexDict = IndexDict()

# grcwa

DEG_TO_RAD = np.pi / 180

# Truncation order (actual number might be smaller)
nG = 10
# lattice constants
L1 = [1,0] # 1 um
L2 = [0,1]
# frequency and angles
theta = 0 * DEG_TO_RAD
phi = 0.
# wls = np.linspace(5, 20, num=300, endpoint=True) # sweep from 5 um to 20 um
freqs = 1 / wv_sweep
Qabs = np.inf
freqcmps = freqs*(1+1j/2/Qabs)
Nx = 100
Ny = 100
Np = 100     # number of discrete layers for sphere

Rs = np.zeros_like(freqs)
Ts = np.zeros_like(freqs)

epgrid = None

for i in range(len(freqs)):
    ######### setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta,phi,verbose=0)
    wavelength = wv_sweep[i]
    index_dict = IndexDict.createIndexDict(['air', 'sio2', 'hfo2', 'ag', 'ti', 'si'], wavelength, '+')
    for material, thickness, type in dev_structure:
        if type == "slab":
            obj.Add_LayerUniform(thickness, index_dict[material])
            print(f'Uniform layer added: {material} {thickness}')
        elif type == "honeycomb":
            epgrid = honeycomb_lattice(obj,Nx,Ny,Np,index_dict[material],thickness)
        else:
            raise NotImplementedError

    obj.Init_Setup()

    obj.GridLayer_geteps(epgrid)

    # planewave excitation
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    # compute reflection and transmission
    R,T= obj.RT_Solve(normalize=1)
    # print('R=',R,', T=',T,', R+T=',R+T)
    Rs[i] = np.real(R)
    Ts[i] = np.real(T)

plt.plot(wv_sweep, Rs)
plt.title('R')
plt.show()

plt.plot(wv_sweep, Ts)
plt.title('T')
plt.show()

As = 1 - Rs - Ts
plt.plot(wv_sweep, As)
plt.title('A')
plt.show()


