import pandas as pd
import numpy as np

class IndexDict:
    def __init__(self):
        self.n_dict = {
            'sio2': pd.read_csv('Herguedas-SiO1.92n.csv'),
            'hfo2': pd.read_csv('BrightHfO2n.csv'),
            'ag': pd.read_csv('Ciesielski-Agn.csv'),
            'ti': pd.read_csv('OrdalTin.csv'),
            'si': pd.read_csv('ShkondinSin.csv')
        }

        self.k_dict = {
            'sio2': pd.read_csv('Herguedas-SiO1.92k.csv'),
            'hfo2': pd.read_csv('BrightHfO2k.csv'),
            'ag': pd.read_csv('Ciesielski-Agk.csv'),
            'ti': pd.read_csv('OrdalTik.csv'),
            'si': pd.read_csv('ShkondinSik.csv')
        }
    
    def interp_wavelength(self, df, wavelength_um, n_or_k):
        """Linear interpolation of n or k of interested wavelength"""
        return np.interp(wavelength_um, df['wl'], df[n_or_k])

    def interp_permittivity(self, material, wavelength_um, k_sign):
        """Use interpolated n and k to calculate permittivity"""
        
        if material == 'air':
            return 1.0 + 0.0j
        
        n = self.interp_wavelength(self.n_dict[material], wavelength_um, 'n')
        k = self.interp_wavelength(self.k_dict[material], wavelength_um, 'k')
        
        if k_sign == '+':
            return (n + 1j*k) ** 2
        elif k_sign == '-':
            return (n - 1j*k) ** 2
        else:
            raise ValueError("k_sign has to be either '+' or '-'")
    
    def createIndexDict(self, materials, wavelength_um, k_sign):
        index_dict = {}
        for material in materials:
            index_dict[material] = self.interp_permittivity(material, wavelength_um, k_sign)
        return index_dict