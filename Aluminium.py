import numpy as np
import matplotlib.pyplot as plt
from lambwaves import Lamb
import warnings

# You can obtain the values of c_L and c_S and an approximate value for
# c_R (if v > 0.3) from the material's mechanical properties by using 
# the following equations:

E = 68.9e9          # E = Young's modulus, in Pa.
p = 2700            # p = Density (rho), in kg/m3.
v = 0.33            # v = Poisson's ratio (nu).

c_L = np.sqrt(E*(1-v) / (p*(1+v)*(1-2*v)))
c_S = np.sqrt(E / (2*p*(1+v)))

c_R = c_S * ((0.862+1.14*v) / (1+v))

# Example: A 10 mm aluminum plate.

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    alum = Lamb(thickness=1.5, #en mm
                nmodes_sym=5, 
                nmodes_antisym=5, 
                fd_max=6000, 
                vp_max=15000, 
                c_L=c_L, 
                c_S=c_S, 
                c_R=c_R,
                material='Aluminum')

# Plot phase velocity, group velocity and wavenumber.

alum.plot_phase_velocity(cutoff_frequencies = False)
alum.plot_group_velocity(cutoff_frequencies = False)
alum.plot_wave_number()


plt.show()
