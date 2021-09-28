import numpy as np
from matplotlib import pyplot as plt 
from scipy import signal
import scipy.optimize as optimize
import math

def get_relative_x_m(x_px):
    return (x_px-origin_px)*meter_per_px

def get_duplicate_indeces(arr):
    dups = []
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]:
            dups.append(i)
    return dups

pos_file = 'raw_data/001/x_pos_pixels.csv'
time_file = 'raw_data/001/times.csv'

ball_radius_px = 35
ball_radius_m = 0.06477
meter_per_px = ball_radius_m/ball_radius_px

origin_px = 1002.8
#origin_py = 786.8

first_n = 18000
length = 1.49

T_error = 0.07
pos_err = 0.05
yerror, xerror = pos_err, 0.001

x_px = np.genfromtxt(pos_file, delimiter=',')[:first_n]
x = get_relative_x_m(x_px)
theta = np.asarray([math.asin(x_curr/length) for x_curr in x])
time = np.genfromtxt(time_file, delimiter=',')[:first_n]
duplicate_indeces = get_duplicate_indeces(theta)
theta = np.delete(theta, duplicate_indeces)
time = np.delete(time, duplicate_indeces)

amp_pos_ind = signal.argrelextrema(theta, np.greater, order = 5)
amp_neg_ind = signal.argrelextrema(theta, np.less, order = 5)

amp_pos = theta[amp_pos_ind]
t_amp_pos = time[amp_pos_ind]
#t_amp_pos = t_amp_pos-t_amp_pos[0] #shift the data so its starts at t=0
amp_neg = theta[amp_neg_ind]
t_amp_neg = time[amp_neg_ind]

period_neg = np.diff(t_amp_neg, 1)
period_pos = np.diff(t_amp_pos, 1)
amp_all = np.concatenate((amp_neg[:-1], amp_pos[:-1][::-1]))
periods = np.concatenate((period_neg, period_pos[::-1]))

amp_per_good = []
i = 0
while i < len(amp_all):
    j = i+1
    while j < len(amp_all) and abs(amp_all[j] - amp_all[i]) < 0.01:
        amp_per_good.append(False)
        j += 1
    i = j-1
    if periods[i] < 3:
        amp_per_good.append(True)
    else:
        amp_per_good.append(False)
    i += 1
    

amp_all = amp_all[amp_per_good]
periods = periods[amp_per_good]

add_amps_neg = [-1.5, -1.2, -0.9, -0.8, -0.72, -0.65]
add_per_neg = [2.54, 2.52, 2.53, 2.51, 2.515, 2.5]

add_amps_pos = [0.65, 0.76, 0.85, 1.15, 1.39]
add_per_pos = [2.51, 2.52, 2.49, 2.51, 2.52]

amp_all = np.concatenate((np.concatenate((add_amps_neg, amp_all)), add_amps_pos), axis = None)
periods = np.concatenate((np.concatenate((add_per_neg, periods)), add_per_pos), axis = None)

# n = 15
# amp_all = amp_all[n:-n]
# periods = periods[n:-n]



def decay_sin(t,a,tau,T,phi):
    return a*np.exp(-t/tau)*np.cos(2*np.pi*t/T+phi)

def decay(t,a,tau):
    return a*np.exp(-t/tau)

def calculate_q_numerically(amplitudes):
    Q = 0
    for amp in amplitudes:
        if amp < math.exp(-math.pi)*init_amp:
            break
        Q += 1
    return Q

def plot_amp_vs_time():
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.plot(t_amp_pos, amp_pos)
    plt.errorbar(t_amp_pos, amp_pos, yerr=yerror, xerr=xerror, errorevery=10, ecolor="red")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (rad)")
    plt.title("Amplitude VS Time")

    plt.subplot(1, 2, 2)
    n = 1000
    plt.plot(time[:n], theta[:n], alpha=0.5)
    plt.plot(t_amp_pos[:17], amp_pos[:17])

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (rad)")
    plt.title("Amplitude VS Time With Position")

    plt.tight_layout()
    plt.show()

def plot_exp_decay_amp():
    
    #print("T error: ", np.std(np.diff(t_amp_pos, 1)))
    popt, pcov = optimize.curve_fit(decay, t_amp_pos, amp_pos)
    
    a = popt[0]
    tau=popt[1]
    print(a, tau)
    
    u_a = pcov[0,0]**(0.5)
    u_tau=pcov[1,1]**(0.5)

    print(u_tau)

    start, stop = min(time), max(time)
    xs = np.arange(start,stop,(stop-start)/1000) 
    curve = decay(xs, a, tau)
    
    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.plot(xs,curve)
    plt.plot(t_amp_pos, amp_pos)
    plt.errorbar(t_amp_pos, amp_pos, yerr=yerror, xerr=xerror, errorevery=10)
    

    plt.text(0.25, 0.6, "y = 0.418*exp(-x/188)", transform=ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (rad)")
    plt.title("Exponential Decay Of Amplitude")
    
    residual=amp_pos-decay(t_amp_pos, a, tau)
    zeroliney=[0,0]
    zerolinex=[start,stop]

    plt.subplot(1, 2, 2)
    plt.errorbar(t_amp_pos,residual,yerr=yerror,xerr=xerror,fmt=".", errorevery=10)
    plt.plot(zerolinex,zeroliney)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Residual (rad)")
    plt.title("Residuals Of The Exponential Fit")
    plt.tight_layout()
    plt.show()

def plot_pos_vs_time():
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    show_n = 100
    plt.errorbar(time[:show_n], theta[:show_n], yerr=yerror, xerr=xerror, errorevery=4, ecolor="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("N = 100")

    plt.subplot(1, 3, 2)
    show_n = 1000
    plt.plot(time[:show_n], theta[:show_n])
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("N = 1000")
    
    plt.subplot(1, 3, 3)
    show_n = 10000
    plt.plot(time[:show_n], theta[:show_n])
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("N = 10000")

    plt.tight_layout()
    plt.show()

def power_series(amp, A, B, C):
    return A + B*amp + C*amp**2

def plot_period_vs_amplitude():

    popt, pcov = optimize.curve_fit(power_series, amp_all, periods)
    
    a = popt[0]
    b = popt[1]
    c = popt[2]
    print(a, b, c)

    au = pcov[0][0]**0.5
    bu = pcov[1][1]**0.5
    cu = pcov[2][2]**0.5
    print(au, bu, cu)

    start, stop = min(amp_all), max(amp_all)
    xs = np.arange(start,stop,(stop-start)/75) 
    curve = power_series(xs, a, b, c)

    f = plt.figure(figsize=(7, 3))
    ax = f.add_subplot(1, 1, 1)
    plt.plot(amp_all, periods, alpha=0.5)
    plt.errorbar(amp_all, periods, yerr=T_error, xerr=pos_err, errorevery=4, ecolor="red", alpha=0.5)
    plt.ylim((2, 3))
    
    plt.xlabel("Amplitude (rad)")
    plt.ylabel("Period (s)")
    plt.title("Amplitude VS Period")
    
    plt.tight_layout()
    plt.show()

    f = plt.figure(figsize=(9, 3))
    ax = f.add_subplot(1, 2, 1)
    plt.plot(amp_all, periods, alpha=0.2)
    plt.errorbar(amp_all, periods, yerr=T_error, xerr=pos_err, errorevery=5, ecolor="red", alpha=0.2)
    plt.ylim((2, 3))
    plt.text(0.2, 0.7, "y = 2.47 - 0.008*x + 0.15*x^2", transform=ax.transAxes)
    plt.plot(xs, curve, color="purple")
    
    plt.xlabel("Amplitude (rad)")
    plt.ylabel("Period (s)")
    plt.title("Amplitude VS Period")

    residual=periods-power_series(amp_all, a, b, c)
    zeroliney=[0,0]
    zerolinex=[start,stop]

    min_swing = math.sqrt(0.022/c)
    print("C:", c)
    print("Min swing:", min_swing)

    plt.subplot(1, 2, 2)
    plt.errorbar(amp_all,residual,yerr=T_error,xerr=xerror,fmt=".", errorevery=2)
    plt.plot(zerolinex,zeroliney)
    plt.ylim(-0.3, 0.3)
    plt.xlabel("Amplitude (rad)")
    plt.ylabel("Residual (s)")
    plt.title("Residuals Of The Power Series Fit")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_period_vs_amplitude()