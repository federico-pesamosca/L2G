import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.interpolate import interp1d

def evaluate_temperature(Told, q, deltaT, time_step):
    # Define q_func 
    def q_func(t):
        idx = np.where(t <= time_step)[0]
        if len(idx) == 0:
            return q[-1]
        return q[idx[0]]
    # Material properties W and SS
    def kk_w(T): return 174.9274 - 0.1067 * T + 5.0067e-5 * T**2 - 7.8349e-9 * T**3
    def cp_w(T): return 128.308 + 3.2797e-2 * T - 3.4097e-6 * T**2
    def rho_w(T): return 1000 * (19.3027 - 2.3786e-4 * T - 2.2448e-8 * T**2)

    def kk_ss(T): return 1.502e-2 * T + 13.98
    def cp_ss(T): return 462.69 + 0.520265 * T - 1.7117e-3 * T**2 + 3.3658e-6 * T**3 - 2.1958e-9 * T**4
    def rho_ss(T): return -6e-5 * T**2 - 0.3977 * T + 7939.3
    
    # Constants and parameters
    Trange = np.arange(20, 801)
    kk_ss_avg = np.mean(kk_ss(Trange))
    rho_ss_avg = np.mean(rho_ss(Trange))
    cp_ss_avg = np.mean(cp_ss(Trange))

    kk_avg = np.mean(kk_w(Trange))
    rho_avg = np.mean(rho_w(Trange))
    cp_avg = np.mean(cp_w(Trange))
    
    LL_tot = 0.011 #0.035
    LL_w = 0.006 #0.015
    LL_ss = LL_tot - LL_w

    nn = 20 #instead of 108, too many
    xx = np.linspace(0, LL_tot, nn)
    dx = xx[1] - xx[0]
    xx1 = np.arange(0, LL_w + dx, dx)
    ni = len(xx1)
    xx2 = np.arange(LL_w + dx, LL_tot + dx, dx)

    dt = deltaT
    #tend = 83 #for s105015 case
    #tend=400 #for s105049 case
    tend=149 #for s105081
    time = np.arange(0, tend + dt, dt)
    
    tau = 1270
    Tw = 100
    hh = 132.5
    kkeq = (kk_ss_avg * LL_ss + kk_avg * LL_w) / LL_tot
    Bi = hh * LL_tot / kkeq

    # Initial condition
    T_0 = Tw
    if Told is None:
        Tnew = np.ones(nn) * T_0
    else:
        Tnew = Told
    
    # central difference discretization
    alpha = np.where(xx <= LL_w, kk_avg / (rho_avg * cp_avg), kk_ss_avg / (rho_ss_avg * cp_ss_avg))
    aa = alpha * dt / dx**2
    inf = np.concatenate([(-aa[1:]).reshape(-1), [0]])
    main = np.concatenate([ (1 + 2 * aa).reshape(-1), [1] ])
    sup = np.concatenate([[0], (-aa[:-1]).reshape(-1)])
    AA = diags([inf, main, sup], [-1, 0, 1], shape=(nn, nn)).tocsc()
    
    # Boundary conditions
    AA[0, 0] = 1
    AA[0, 1] = -1
    
    AA[ni-1, ni-2] = -kk_avg / kk_ss_avg
    AA[ni-1, ni-1] = 1 + kk_avg / kk_ss_avg
    AA[ni-1, ni] = -1
    
    AA[-1, -2] = -1
    AA[-1, -1] = 1 + hh * dx / kk_ss_avg

    # Inverse of the matrix
    BB = np.linalg.inv(AA.toarray())

    Tsurf = np.zeros(len(time))
    Tsurf_sampled = np.zeros(len(time_step))
    Tprofiles = np.zeros((len(xx), len(time_step)))

    flux = np.zeros(len(time))
    
    for ii in range(1, len(time)):
        bb = Tnew
        flux[ii] = q_func(time[ii])
        
        bb[0] = flux[ii] * dx / kk_avg
        bb[ni-1] = 0
        bb[-1] = hh * dx / kk_ss_avg * Tw
        
        TT = BB @ bb
        
        Tsurf[ii] = TT[0]
        
        # Sampling at specific time steps
        match_idx = np.where(np.abs(time[ii] - time_step) < 0.5)
        # match_idx = np.where(np.abs(time[ii] - time_step) < 1e-3)
        # print(match_idx, time[ii] - time_step)
        if match_idx[0].size > 0:
            Tsurf_sampled[match_idx] = Tsurf[ii]
            Tprofiles[:, match_idx] = TT.reshape((nn, 1, 1))
        
        Tnew = TT

    return Tnew, time, xx, Tsurf_sampled, Tsurf, Tprofiles

#integrate the results in smiter output

import l2g.mesh

import sys
if not len(sys.argv) > 2:
    print('No path provided to a MED file!')
    sys.exit(1)
med_file_path = sys.argv[1]
mesh = l2g.mesh.Mesh(med_file_path)

cell_measurements = mesh.getCellMeasurements()

field_name = "q_perp"

field_iterations = mesh.getAllFieldIterations(field_name)

times = np.array([_[1] for _ in field_iterations])
time_step = 1

F = []

for i, time in field_iterations:
    mesh.setIndex(i)

    field = mesh.getField(field_name)
    F.append(field)

number_of_time_steps = len(times)
number_of_cells = len(F[0])

Twater=100

T = Twater*np.ones(F[0].shape, F[0].dtype) #initialize at Tw=100

temperature = Twater*np.ones((len(times), F[0].shape[0]), F[0].dtype) #initialize at Tw=100


for i in range(number_of_cells):
    # print(i)
    # Create the Q on this cell
    q = np.array([F[j][i] for j in range(number_of_time_steps)])

    # Try to evaluate if there is any energy on the cell and accordingly skip it if there
    # is no heat load (or it's shadowed.)

    # Calculate energy on this particular cell

    diff_t = np.diff(times)
    energy = np.sum(0.5 * (q[1:] + q[:-1]) * diff_t) * cell_measurements[i] * 1e-6

    if energy < 1e-5: #1e-2 for small q''
      continue

    _, _, _, Tsurf_sampled, Tsurf, _ = evaluate_temperature(None, q, time_step, times)

    T[i] = Tsurf[-1]

    temperature[:, i] = Tsurf_sampled


# Now we add field only at last time
mesh.setIndex(0)
mesh.setTime(times[-1])

mesh.addField("Tsurf", T)
mesh.writeFields()

for i in range(len(times)):
    mesh.setIndex(i)
    mesh.setTime(times[i])

    mesh.addField("T", temperature[i])
    mesh.writeFields()

