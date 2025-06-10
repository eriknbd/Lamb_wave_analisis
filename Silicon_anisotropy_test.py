import numpy as np
import matplotlib.pyplot as plt
from lambwaves import Lamb
from lambmath import wave_speeds
import warnings

f = 1e3     #kHz
b = 0.500   #mm

E = 170e9           # E = Young's modulus, in Pa.
p = 2330            # p = Density (rho), in kg/m3.
v = 0.064           # v = Poisson's ratio (nu).
c_L, c_S, c_R = wave_speeds(E, v, p)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    si110 = Lamb(thickness=b, #en mm
                nmodes_sym=5, 
                nmodes_antisym=5, 
                fd_max=6000, 
                vp_max=15000, 
                c_L=c_L, 
                c_S=c_S, 
                c_R=c_R,
                material='Aluminum')


E = 130e9           # E = Young's modulus, in Pa.
p = 2330            # p = Density (rho), in kg/m3.
v = 0.3             # v = Poisson's ratio (nu).
c_L, c_S, c_R = wave_speeds(E, v, p)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    si100 = Lamb(thickness=b, #en mm
                nmodes_sym=5, 
                nmodes_antisym=5, 
                fd_max=6000, 
                vp_max=15000, 
                c_L=c_L, 
                c_S=c_S, 
                c_R=c_R,
                material='Aluminum')

vf110 = si110.vp_antisym['A0'](f*b)
vf100 = si100.vp_antisym['A0'](f*b)
vf_chage = vf110 - vf100

v_sonido = 343          #Velocidad del sonido en el aire [m/s]
ang_110 = np.arcsin(v_sonido/vf110)
ang_100 = np.arcsin(v_sonido/vf100)
ang_change = ang_110 - ang_100

print(f"\nFrecuency X Thickness = {f*b} kHz·mm")
print("----------Velocity change----------")
print(f'[110] -> {vf110.round(2)}, [100] -> {vf100.round(2)}')
print(f'Diference: {round(abs(vf_chage), 2)} m/s')
print(f'           {round(abs(vf_chage/max([vf110, vf100]))*100, 2)} %')

print("--------angle change (Deg)---------")
print(f'[110] -> {round(ang_110*180/np.pi, 2)}, [100] -> {round(ang_100*180/np.pi, 2)}')
print(f'Diference: {round(abs(ang_change*180/np.pi), 2)}')
print(f'           {round(abs(ang_change/max([ang_110, ang_100]))*100, 2)} %')