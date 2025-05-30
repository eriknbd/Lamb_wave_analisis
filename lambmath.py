import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import (hilbert, butter, filtfilt,
                          correlate, correlation_lags)
from scipy.optimize import brentq


def fft_gen(time_array, signal):

    dt = time_array[1] - time_array[0] 
    N = len(time_array)

    window = np.hanning(len(signal))
    signal_windowed = signal * window

    Y = np.fft.fft(signal_windowed)
    f = np.fft.fftfreq(N, dt)

    idx = f >= 0
    f_pos = f[idx]
    Y_pos = Y[idx]

    mag = (2.0 / N) * np.abs(Y_pos)
    return f_pos, mag


def max_frecuency(data_in, plot = False, db = True):

    time = data_in.T[0]
    signal = data_in.T[1:]


    if signal.shape[0] == 1:
        f_pos , mag = fft_gen(time, signal)

    else:
        f_pos = fft_gen(time, signal[0])[0]
        mag = []
        for n in range(len(signal)):
            mag.append(fft_gen(time, signal[n])[1])
        mag = np.mean(mag, axis = 0)

    f_max = f_pos[np.argmax(mag)]

    if plot:
        plt.figure(figsize=(12,4))
        if db:
            plt.plot(f_pos*10**-6, 20*np.log10(mag))
            plt.ylabel("Amplitud [dB]")
        else:
            plt.plot(f_pos*10**-6, mag)
            plt.ylabel("Amplitud")
        plt.title("FFT de la señal")
        plt.xlabel("Frecuencia [MHz]")
        plt.grid(True)
        plt.xlim(0, 1.5)  
        plt.show()

    return f_max    


def bandpass(data_in, bp_w, bp_c = None, bp_order = 4):

    time = data_in.T[0]
    signals = data_in.T[1:]

    if bp_c == None:
        bp_c = max_frecuency(data_in)

    lowcut = bp_c - bp_w/2   # Hz
    highcut= bp_c + bp_w/2   # Hz

    dt = time[1] - time[0]  
    fs = 1.0 / dt

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(bp_order , [low, high], btype='band')

    new_signals = [filtfilt(b, a, signal) for signal in signals]


    return np.concat((np.array([time]), new_signals)).T


def cc_tau(time, signal_1, signal_2):

    dt = time[1] - time[0]
    fs = 1.0 / dt
    
    corr = correlate(signal_2, signal_1, mode='full')          
    lags = correlation_lags(len(signal_2), len(signal_1), mode='full')
    lag_index = np.argmax(np.abs(corr))
    tau = lags[lag_index] / fs

    return tau


def cc_group_velocity(data_in, d, bp_w = None, bp_c = None, bp_order = 4, plot = False, offset = None, title = None):

    if bp_w != None: 
        data_in = bandpass(data_in, bp_w, bp_c, bp_order)

    time = data_in.T[0]
    signals = data_in.T[1:]
    n = signals.shape[0]

    if offset == None:
        offset = signals[0].max()

    env = [np.abs(hilbert(signal)) for signal in signals]

    t = [0]

    for i in range(n-1):
        t.append(t[i]+cc_tau(time, env[i], env[i+1]))

    t = np.array(t)
    lr = linregress(t, d)

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        if title != None:
            ax[0].set_title(title, 
                         fontsize=14,
                        fontweight='bold',
                        loc='left',
                        pad=10)
        ax[0].set_xlabel("time [µs]")
        ax[0].set_yticks([])
        ax[1].set_xlabel("t [µs]")
        ax[1].set_ylabel("d [m]")

        for i in range(n):
            ax[0].plot((time- t[i])*10**6, signals[i] - i*offset)
            ax[0].plot((time- t[i])*10**6, env[i] - i*offset, c = "red")
        
        ax[1].scatter(t*10**6, d)
        ax[1].plot(t*10**6, t*lr[0] +lr[1], c = "red")
        fig.tight_layout()
        plt.show()

    plt.show()
    
    return np.array([lr[0], lr[1], lr[2]**2])


def cc_group_velocity_all(data_in, d,
                          bp_w=None, bp_c=None, bp_order=4,
                          plot=False, offset=None, title=None):

    # ----------- pre-procesado ---------------------
    if bp_w is not None:
        data_in = bandpass(data_in, bp_w, bp_c, bp_order)

    time    = data_in.T[0]
    signals = data_in.T[1:]
    n       = signals.shape[0]

    if offset is None:
        offset = signals[0].max()

    env = [np.abs(hilbert(signal)) for signal in signals]   # envolventes

    # ---------------- correlaciones cruzadas --------------------------
    taus, dists = [], []
    for i in range(n):
        for j in range(n):
            if i == j or i < j:
                continue
            tau_ij = cc_tau(time, env[i], env[j])   # retraso j respecto a i
            taus.append(tau_ij)
            dists.append(d[j] - d[i])              # distancia j-i

    taus, dists = np.array(taus), np.array(dists)

    lr = linregress(taus, dists)                   # pendiente = velocidad
    vg      = lr.slope
    y0      = lr.intercept
    rsquare = lr.rvalue**2

    # -------------------- gráfica --------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        if title:
            ax[0].set_title(title, fontweight='bold', loc='left', pad=10)
        ax[0].set_xlabel("time [µs]")
        ax[0].set_yticks([])
        ax[1].set_xlabel("τ [µs]")
        ax[1].set_ylabel("Δd [m]")

        # Para ilustrar: “alineamos” cada traza usando su τ medio respecto a la 0
        tau_ref = d/vg
        
        for i in range(n):
            ax[0].plot((time - tau_ref[i]) * 1e6, signals[i] - i*offset)
            ax[0].plot((time - tau_ref[i]) * 1e6, env[i]     - i*offset, c="red")

        ax[1].scatter(taus*1e6, dists)
        ax[1].plot(taus*1e6, vg*taus + y0, c="red")
        fig.tight_layout()
        plt.show()

    return np.array([vg, y0, rsquare])


def cc_phase_velocity(data_in, d, type_dist = "inv", bp_w = None, bp_c = None, plot = False, offset = None, title = None):

    if bp_w != None:
        data_in = bandpass(data_in, bp_w, bp_c)
    
    time = data_in.T[0]
    signals = data_in.T[1:]

    if offset == None:
        offset = signals[0].max()

    if type_dist == "inv":
        d = d[14:]
        signals = signals[14:]
    elif type_dist == "dir":
        d = d[:21]
        signals = signals[:21]
    n = signals.shape[0]

    t = [0]
    for i in range(n-1):
        t.append(t[i] + cc_tau(time, signals[i], signals[i+1]))
    
    t  = np.array(t)
    lr = np.array(linregress(t, d))

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        if title != None:
            ax[0].set_title(title, 
                         fontsize=14,
                        fontweight='bold',
                        loc='left',
                        pad=10)
        ax[0].set_xlabel("t [µs]")
        ax[0].set_yticks([])
        ax[1].set_xlabel("t [µs]")
        ax[1].set_ylabel("d [m]")

        for i in range(n):
            ax[0].plot((time - t[i])*10**6, signals[i]- i*signals[1].max())
        
        ax[1].scatter(t*10**6, d)
        ax[1].plot(t*10**6, t*lr[0] +lr[1], c = "red")
        fig.tight_layout()
        plt.show()


    return np.array([lr[0], lr[1], lr[2]**2])


def full_analisis(data_in, d, type_dist, bp_c = None, bp_w = None, plot = True, print_data = True, round_to = 4, title = None):

    f_max = max_frecuency(data_in, plot = plot)
    if print_data:
        print(f'Mean frecuency max = {round(f_max*10**-3, round_to)} kHz.')

    vg = cc_group_velocity(data_in, d, bp_c = bp_c, bp_w = bp_w, plot = plot, title= title)
    if print_data:
        print(f'vg = {round(vg[0], round_to)} m/s; Intercept = {round(vg[1], round_to)}; r2 = {round(vg[2], round_to)}.')

    vf = cc_phase_velocity(data_in, d, bp_c = bp_c, bp_w = bp_w, plot = plot, title= title, type_dist = type_dist)
    if print_data:
        print(f'vf = {round(vf[0], round_to)} m/s; Intercept = {round(vf[1], round_to)}; r2 = {round(vf[2], round_to)}.')

    return f_max, vg, vf


def amplitudes(data_in, d = None, bp_w = None, bp_c = None, plot = True):

    if bp_w != None:
        data_in = bandpass(data_in, bp_w, bp_c)

    signals = data_in.T[1:]

    A = np.array([np.abs(signal).max() for signal in signals])

    if plot:
        plt.figure(figsize = (6,4))
        plt.scatter(d, np.log(A/A.max()))
        plt.show()

    return A

def _rayleigh_ratio(v):

    k = (1 - 2*v) / (2*(1 - v))

    f = lambda x: x**6 - 8*x**4 + 8*(3 - 2*k)*x**2 - 16*(1 - k)

    return brentq(f, 1e-9, 1 - 1e-9, maxiter=100, xtol=1e-12)

def wave_speeds(E, v, p):
    if not (-1.0 < v < 0.5):
        raise ValueError("ν debe estar en (-1, 0.5).")
    if E <= 0 or p <= 0:
        raise ValueError("E y ρ deben ser positivos.")

    # ondas volumétricas exactas
    c_L = np.sqrt(E*(1 - v) / (p*(1 + v)*(1 - 2*v)))
    c_S = np.sqrt(E / (2*p*(1 + v)))

    # onda superficial exacta (resolviendo la ecuación)
    xi   = _rayleigh_ratio(v)
    c_R  = xi * c_S
    return c_L, c_S, c_R



def cros_time(time, signal, thr, min_time):
    dt = time[1] - time[0]
    n_pass = np.ceil(min_time/dt)
    idx_valid = np.array([])

    if thr == None:
        thr = signal.max()*(2/5)

    while idx_valid.size == 0:
        higher = signal > thr

        counts = np.convolve(higher.astype(int),
                np.ones((n_pass.astype(int))),
                mode='valid')

        idx_valid = np.argwhere(counts == n_pass)
        thr = thr*9/10        

    return time[idx_valid[0]][0]

def threshold_tau(time, signal_1, signal_2, thr  = None, min_time = None):
    if min_time == None:
        min_time = (time[-1] - time[0])/642

    tau = cros_time(time, signal_2, thr, min_time) - cros_time(time, signal_1, thr, min_time)
    return tau


def threshold_group_velocity(data_in, d, bp_w = None, bp_c = None, bp_order = 4, plot = False, offset = None, title = None):

    if bp_w != None: 
        data_in = bandpass(data_in, bp_w, bp_c, bp_order)

    time = data_in.T[0]
    signals = data_in.T[1:]
    n = signals.shape[0]

    if offset == None:
        offset = signals[0].max()

    env = [np.abs(hilbert(signal)) for signal in signals]

    t = [0]

    for i in range(n-1):
        t.append(t[i]+threshold_tau(time, env[i], env[i+1]))

    t = np.array(t)
    lr = linregress(t, d)

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        if title != None:
            ax[0].set_title(title, 
                         fontsize=14,
                        fontweight='bold',
                        loc='left',
                        pad=10)
        ax[0].set_xlabel("time [µs]")
        ax[0].set_yticks([])
        ax[1].set_xlabel("t [µs]")
        ax[1].set_ylabel("d [m]")

        for i in range(n):
            ax[0].plot((time- t[i])*10**6, signals[i] - i*offset)
            ax[0].plot((time- t[i])*10**6, env[i] - i*offset, c = "red")
        
        ax[1].scatter(t*10**6, d)
        ax[1].plot(t*10**6, t*lr[0] +lr[1], c = "red")
        fig.tight_layout()
        plt.show()

    plt.show()
    
    return np.array([lr[0], lr[1], lr[2]**2])


def threshold_group_velocity_all(data_in, d,
                          bp_w=None, bp_c=None, bp_order=4,
                          plot=False, offset=None, title=None):

    # ----------- pre-procesado ---------------------
    if bp_w is not None:
        data_in = bandpass(data_in, bp_w, bp_c, bp_order)

    time    = data_in.T[0]
    signals = data_in.T[1:]
    n       = signals.shape[0]

    if offset is None:
        offset = signals[0].max()

    env = [np.abs(hilbert(signal)) for signal in signals]   # envolventes

    # ---------------- correlaciones cruzadas --------------------------
    taus, dists = [], []
    for i in range(n):
        for j in range(n):
            if i == j or i < j:
                continue
            tau_ij = threshold_tau(time, env[i], env[j])   # retraso j respecto a i
            taus.append(tau_ij)
            dists.append(d[j] - d[i])              # distancia j-i

    taus, dists = np.array(taus), np.array(dists)

    lr = linregress(taus, dists)                   # pendiente = velocidad
    vg      = lr.slope
    y0      = lr.intercept
    rsquare = lr.rvalue**2

    # -------------------- gráfica --------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        if title:
            ax[0].set_title(title, fontweight='bold', loc='left', pad=10)
        ax[0].set_xlabel("time [µs]")
        ax[0].set_yticks([])
        ax[1].set_xlabel("τ [µs]")
        ax[1].set_ylabel("Δd [m]")

        # Para ilustrar: “alineamos” cada traza usando su τ medio respecto a la 0
        tau_ref = d/vg
        
        for i in range(n):
            ax[0].plot((time - tau_ref[i]) * 1e6, signals[i] - i*offset)
            ax[0].plot((time - tau_ref[i]) * 1e6, env[i]     - i*offset, c="red")

        ax[1].scatter(taus*1e6, dists)
        ax[1].plot(taus*1e6, vg*taus + y0, c="red")
        fig.tight_layout()
        plt.show()

    return np.array([vg, y0, rsquare])