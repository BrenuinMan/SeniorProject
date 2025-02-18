#%%
""" Topology optimization of reflection of a single patterned layers ."""
""" Nlopt is needed """
# Try more layers
# Sweep angle

import grcwa
grcwa.set_backend('autograd')

import autograd.numpy as np
from autograd import grad
from IndexDict import IndexDict
import matplotlib.pyplot as plt 

nm_to_um = 1e-3

dev_structure = [
    ('air',0.0*nm_to_um),
    ('sio2',230.0*nm_to_um),
    ('hfo2',485.0*nm_to_um),
    ('sio2',688.0*nm_to_um),
    ('hfo2',13.0*nm_to_um),
    ('sio2',73.0*nm_to_um),
    ('hfo2',34.0*nm_to_um),
    ('sio2',54.0*nm_to_um),
    ('ag',200.0*nm_to_um),
    ('ti',20.0*nm_to_um),
    ('si',750.0*nm_to_um),
    ('air',0.0*nm_to_um)
]
start_wv = 5  # 5 um
end_wv = 20   # 20 um
wv_sweep = np.linspace(start_wv, end_wv, num=100, endpoint=True)

IndexDict = IndexDict()

# Truncation order (actual number might be smaller)
nG = 10
# lattice constants
L1 = [1,0] # 1 um
L2 = [0,1]
# frequency and angles
freqs = 1 / wv_sweep
Qabs = np.inf
freqcmps = freqs*(1+1j/2/Qabs)
theta = np.linspace(0, 89 * np.pi / 180, num=10)
#%%
phi = 0.
# the patterned layer has a griding: Nx*Ny
Nx = 100
Ny = 100

# planewave excitation
planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}

materials = [material for material, _ in dev_structure]

def training_loss(thicknesses, plot=False, title=''):
    loss = np.array(0.)

    Ass = np.zeros((len(theta), len(freqs)))

    for i in range(len(freqs)):
        for ti, th in enumerate(theta):
            obj = grcwa.obj(nG,L1,L2,freqcmps[i],th,phi,verbose=0)
            wavelength = wv_sweep[i]
            index_dict = IndexDict.createIndexDict(['air', 'sio2', 'hfo2', 'ag', 'ti', 'si'], wavelength, '+')
            for j in range(len(materials)):
                obj.Add_LayerUniform(thicknesses[j], index_dict[materials[j]])

            obj.Init_Setup()

            # planewave excitation
            planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
            obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

            # compute reflection and transmission
            R,T= obj.RT_Solve(normalize=1)
            A = 1 - R - T
            A = np.real(A)
            if plot:
                Ass[ti, i] = A
            if 8 <= wavelength <= 13:
                pass
                loss += ((A - 1)**2)
            else:
                loss += (A**2)

    if plot:
        plt.clf()
        for ti in range(len(theta)):
            plt.plot(wv_sweep, Ass[ti, :])
        plt.title('A')
        # plt.show()
        if title != '':
            plt.savefig(title)
    return loss

training_gradient_fun = grad(training_loss)

thicknesses = np.array([thickness for _, thickness in dev_structure])
# thicknesses = np.ones_like(thicknesses) * .01

print("Initial loss:", training_loss(thicknesses, plot=True))
for i in range(100):
    thicknesses -= training_gradient_fun(thicknesses) * 0.001
    if i % 1 == 0:
        print(f"Epoch {i}: ", training_loss(thicknesses, plot=i%10 == 0, title=f"Epoch {i}.png"))
        print(thicknesses)

print("Trained loss:", training_loss(thicknesses, plot=True, title='final.png'))
print(thicknesses)
#%%

# [-3.38546865e-19  3.67212544e-01  3.59436012e-01  8.76011988e-01
# -2.10749357e-01  2.67480220e-01 -1.80607459e-01  2.53924513e-01
#  1.99999224e-01  1.99999549e-02  7.50000007e-01  0.00000000e+00]