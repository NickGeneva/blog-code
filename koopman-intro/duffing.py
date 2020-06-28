'''
Simple duffing simulator code
https://scipython.com/blog/the-duffing-oscillator/
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from scipy.integrate import odeint, quad

# The potential and its first derivative, as callables.
V = lambda x: 0.5 * x**2 * (0.5 * x**2 - 1)
dVdx = lambda x: x**3 - x

def deriv(X, t, gamma, delta, omega, alpha=1.0, beta=-1.0):
    """Return the derivatives dx/dt and d2x/dt2."""

    x, xdot = X
    xdotdot = -x*(beta + alpha*x**2) - delta * xdot + gamma * np.cos(omega*t)
    return xdot, xdotdot

def solve_duffing(tmax, dt_per_period, t_trans, x0, v0, gamma, delta, omega):
    """Solve the Duffing equation for parameters gamma, delta, omega.

    Find the numerical solution to the Duffing equation using a suitable
    time grid: tmax is the maximum time (s) to integrate to; t_trans is
    the initial time period of transient behaviour until the solution
    settles down (if it does) to some kind of periodic motion (these data
    points are dropped) and dt_per_period is the number of time samples
    (of duration dt) to include per period of the driving motion (frequency
    omega).

    Returns the time grid, t (after t_trans), position, x, and velocity,
    xdot, dt, and step, the number of array points per period of the driving
    motion.

    """
    # Time point spacings and the time grid

    period = 2*np.pi/omega
    dt = 2*np.pi/omega / dt_per_period
    step = int(period / dt)
    t = np.arange(0, tmax, dt)
    # Initial conditions: x, xdot
    X0 = [x0, v0]
    X = odeint(deriv, X0, t, args=(gamma, delta, omega))
    idx = int(t_trans / dt)
    return t[idx:], X[idx:], dt, step

if __name__ == '__main__':
    # Set up the motion for a oscillator with initial position
    # x0 and initially at rest.
    x0, v0 = -1, -1
    tmax, t_trans = 15, 0
    omega = 1.0
    gamma, delta = 0.0, 0.5
    dt_per_period = 50

    # Switch fonts
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    rc('text', usetex=True)
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    x_data = []
    xdot_data = []
    t_data = []
    # Solve the equation of motion.
    for x0 in np.arange(-2, 2+1e-8, 0.2):
        for v0 in np.arange(-2, 2+1e-8, 0.2):
            t, X, dt, pstep = solve_duffing(tmax, dt_per_period, t_trans, x0, v0, gamma, delta, omega)
            x, xdot = X.T
            x_data.append(x)
            xdot_data.append(xdot)
            t_data.append(t)
    
            cmap = plt.get_cmap("viridis")
            dotColors = cmap(np.linspace(0,1,len(xdot)))
            # for i in range(0,len(xdot)-1):
            #     ax[0].plot(x[i:i+2], xdot[i:i+2], color=dotColors[i])
            ax.scatter(x, xdot, c=dotColors, s=0.5, alpha=0.8) 

    data = np.stack([np.array(x_data), np.array(xdot_data), np.array(t_data)], axis=2)
    print('data-size', data.shape)
    np.save('duffing.npy', data)