import gradio as gr
import matplotlib.pyplot as plt 
import numpy as np

# Add parent directory to sys.path
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)

DEG_TO_RAD = np.pi / 180

def materialDropdowns(num):
    inputs = []
    for i in range(num):
        inputs.append(gr.Dropdown(["None", "sio2", "hfo2", "ag", "ti", "si"], 
                                  label=f"Material {i}"))
    return inputs

def thicknessDropdowns(num):
    inputs = []
    for i in range(num):
        inputs.append(gr.Number(0.0, label=f"Thickness {i} (nm)"))
    return inputs

def run(*args):
    args_list = list(args)
    angle = args_list.pop()
    mid = len(args_list) // 2
    materials = args_list[:mid]
    thicknesses = args_list[mid:]
    return prc(materials, thicknesses, angle)

def run_opt(*args):
    args_list = list(args)
    angle = args_list.pop()
    mid = len(args_list) // 2
    materials = args_list[:mid]
    thicknesses = args_list[mid:]
    return opt(materials, thicknesses, angle)

def prc(materials, thicknesses, angle):
    from IndexDict import IndexDict
    import numpy as np
    import grcwa

    nm_to_um = 1e-3

    dev_structure = [
        ('air',0.0*nm_to_um),
    ]

    for i in range(len(materials)):
        dev_structure.append((materials[i], thicknesses[i]*nm_to_um))

    dev_structure.append(('air',0.0*nm_to_um))

    start_wv = 5  # 5 um
    end_wv = 20   # 20 um
    wv_sweep = np.linspace(start_wv, end_wv, num=100, endpoint=True)

    IndexDict = IndexDict()

    # grcwa

    # Truncation order (actual number might be smaller)
    nG = 10
    # lattice constants
    L1 = [1,0] # 1 um
    L2 = [0,1]
    # frequency and angles
    theta = angle * DEG_TO_RAD
    phi = 0.
    # wls = np.linspace(5, 20, num=300, endpoint=True) # sweep from 5 um to 20 um
    freqs = 1 / wv_sweep
    Qabs = np.inf
    freqcmps = freqs*(1+1j/2/Qabs)
    Nx = 100
    Ny = 100

    Rs = np.zeros_like(freqs)
    Ts = np.zeros_like(freqs)

    for i in range(len(freqs)):
        ######### setting up RCWA
        obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta,phi,verbose=0)
        wavelength = wv_sweep[i]
        index_dict = IndexDict.createIndexDict(['air', 'sio2', 'hfo2', 'ag', 'ti', 'si'], wavelength, '+')
        for material, thickness in dev_structure:
            obj.Add_LayerUniform(thickness, index_dict[material])

        obj.Init_Setup()

        # planewave excitation
        planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

        # compute reflection and transmission
        R,T= obj.RT_Solve(normalize=1)
        # print('R=',R,', T=',T,', R+T=',R+T)
        Rs[i] = np.real(R)
        Ts[i] = np.real(T)

    # plt.plot(wv_sweep, Rs)
    # plt.title('R')
    # plt.show()

    # plt.plot(wv_sweep, Ts)
    # plt.title('T')
    # plt.show()

    As = 1 - Rs - Ts
    fig = plt.figure()
    plt.plot(wv_sweep, As)
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Absorbance')
    return fig

def opt(materials, thicknesses, angle):
    import grcwa
    grcwa.set_backend('autograd')

    import autograd.numpy as np
    from autograd import grad
    from IndexDict import IndexDict
    import matplotlib.pyplot as plt 

    nm_to_um = 1e-3

    dev_structure = [
        ('air',0.0*nm_to_um),
    ]

    for i in range(len(materials)):
        dev_structure.append((materials[i], thicknesses[i]*nm_to_um))

    dev_structure.append(('air',0.0*nm_to_um))

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
    theta = angle * DEG_TO_RAD
    phi = 0.

    # planewave excitation
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}

    materials = [material for material, _ in dev_structure]

    def training_loss(thicknesses, plot=False, title=''):
        loss = np.array(0.)

        As = np.zeros_like(freqs)

        for i in range(len(freqs)):
            obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta,phi,verbose=0)
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
                As[i] = A
            if 8 <= wavelength <= 13:
                pass
                loss += (A - 1)**2
            else:
                loss += A**2

        if plot:
            plt.plot(wv_sweep, As)
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
        if i % 10 == 0:
            print(f"Epoch {i}: ", training_loss(thicknesses, plot=True, title=f"Epoch {i}.png"))

    print("Trained loss:", training_loss(thicknesses, plot=True, title='final.png'))
    return prc(materials, thicknesses, angle)

num_layers = 10

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            materials = materialDropdowns(num_layers)
        with gr.Column():
            thicknesses = thicknessDropdowns(num_layers)
        with gr.Column():
            with gr.Row():
                theta = gr.Number(0, label=f"Angle (degrees)")
            with gr.Row():
                run_btn = gr.Button(value="Run")
                opt_btn = gr.Button(value='Optimize')
            result = gr.Plot(label="Result", format="png")

    run_btn.click(
        run, inputs=materials + thicknesses + [theta], outputs=[result], api_name=False
    )

    opt_btn.click(
        run_opt, inputs=materials + thicknesses + [theta], outputs=[result], api_name=False
    )

    examples = gr.Examples(
        examples=[['sio2', 'hfo2', 'sio2', 'hfo2', 'sio2', 'hfo2', 'sio2', 'ag', 'ti', 'si',
                   230.0, 485.0, 688.0, 13.0, 73.0, 34.0, 54.0, 200.0, 20.0, 750.0,
                   0]],
        inputs=materials+thicknesses+[theta],
    )

demo.launch()
