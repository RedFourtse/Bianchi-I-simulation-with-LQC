import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import matplotlib as mpl
import sympy as sp

G = 1
rho_crit = 0.41
n = 10 #n != 1 because int not interable (cant put int as str)
gamma = 0.2375 #barbero immrizi parameter approximation
length_gap = 4 * np.sqrt(3) * np.pi * gamma  #for natural units h-bar = G = c = 1, l_gap ^2 or delta squared
a_x = np.linspace(5, 300, n)
a_y = np.linspace(5, 300, n)
a_z = np.linspace(5, 300, n)
"""a_x = np.linspace(11, 32, n)
a_y = np.linspace(11, 33, n)
a_z = np.linspace(12, 34, n)"""

p_phi = 15 #any constant
p_phi_symbol = sp.symbols("p_phi", real = True)
p_1, p_2, p_3 = sp.symbols("p_1 p_2 p_3", real = True)
c_1, c_2, c_3 = sp.symbols("c_1 c_2 c_3", real = True)
#See Page 4
mu_1 = sp.sqrt(abs(p_1) * length_gap / abs(p_2 * p_3))
mu_2 = sp.sqrt(abs(p_2) * length_gap / abs(p_1 * p_3))
mu_3 = sp.sqrt(abs(p_3) * length_gap / abs(p_1 * p_2))
b_1 = mu_1 * c_1
b_2 = mu_2 * c_2
b_3 = mu_3 * c_3
V = sp.sqrt(abs(p_1 * p_2 * p_3))
rho = (p_phi**2) / (2.0 * V**2)
B = p_1 * p_2 * abs(p_3) *sp.sin(b_1) * sp.sin(b_2) + p_1 * abs(p_2) * p_3 * sp.sin(b_1) * sp.sin(b_3) +  abs(p_1) * p_2 *p_3 * sp.sin(b_2) * sp.sin(b_3)
#for p_1, p_2, p_3 > 0
S = B/V**2
C_grav = -V * S/(8*np.pi *G * gamma**2 * length_gap) #3.14 Ahstekar Wilsion Elwing 2009
C_matter = p_phi_symbol**2/(2* V)
C_eff = C_grav + C_matter

state = [p_1, p_2, p_3, c_1, c_2, c_3, p_phi_symbol]
dot_p = [-8 * sp.pi * G * gamma * sp.diff(C_eff, c) for c in [c_1, c_2, c_3]] #{c_i, C_eff} = 8 pi G gamma d(C_eff)/dp_i; 
dot_c = [8 * sp.pi * G * gamma * sp.diff(C_eff, p) for p in [p_1, p_2, p_3]] 
expr = dot_p + dot_c #dot_p_phi = 0
func = sp.lambdify([p_1, p_2, p_3, c_1, c_2, c_3, p_phi_symbol], expr, "numpy")
results = []
for init_a_x, init_a_y, init_a_z in itertools.product(a_x, a_y, a_z):
    init_p = np.array([ np.sign(init_a_x)*abs(init_a_y*init_a_z), np.sign(init_a_y)*abs(init_a_x*init_a_z), np.sign(init_a_z)*abs(init_a_x*init_a_y)])
    init_V = np.sqrt(abs(np.prod(init_p)))
    K = 4 * np.pi * G * gamma**2 * length_gap * p_phi**2 / init_V**2
    if init_p.prod() < (K * init_V/3):
        print("fix: initial triad p")
    init_b_1 = -0.133011 #set to anything
    init_b_2 = -0.133011 #set to anything
    if ((K - np.sin(init_b_1) * np.sin(init_b_2))/(np.sin(init_b_1) + np.sin(init_b_2))) > 1 or ((K - np.sin(init_b_1) * np.sin(init_b_2))/(np.sin(init_b_1) + np.sin(init_b_2))) < -1: 
        print("fix: choose new init_b_1/init_b_2") 
        print()
    init_b_3 = np.arcsin((K - np.sin(init_b_1) * np.sin(init_b_2))/(np.sin(init_b_1) + np.sin(init_b_2)))

    init_b = [init_b_1, init_b_2, init_b_3] 
    #See page 4 for "proof"
    init_mu = np.sqrt(length_gap) * np.abs(init_p) / init_V
    init_c = init_b / init_mu
    #these are for all triad p are positive and sin(b_1) != sin(b_2)
    init_S = (np.sin(init_b[0]) * np.sin(init_b[1])+ np.sin(init_b[1]) * np.sin(init_b[2]) + np.sin(init_b[2]) * np.sin(init_b[0]))
    if init_S < 0: 
        print("fix: b_i not real")

    init_state = np.concatenate((init_p, init_c))
    t=100
    time_span = (0, t) 
    t_eval = np.linspace(time_span[0], time_span[1], t)

    def system(t, state):
        p = state[:3]
        c = state[3:6]
        if not np.all(np.isfinite(state)):
            print("fix:not finite p_i or c_i")
        if np.any(p <= 0):
            print("fix:triad negative")
        derivatives = func(*p, *c, p_phi)
        return np.asarray(derivatives, dtype=float)

    solution = solve_ivp(system, (0, t), init_state, method="DOP853", t_eval = t_eval, rtol = 1e-9, atol = 1e-11)
    if not solution.success:
        print(solution.message)
        continue
    p_data = solution.y[:3]
    c_data = solution.y[3:6]

    p_1_data, p_2_data, p_3_data = p_data
    c_1_data, c_2_data, c_3_data = c_data
    V_data = np.sqrt(p_1_data * p_2_data * p_3_data)
    #a_x, a_y, a_z are the initial scale factors, a_q, a_2, a_3 are the variable ones
    a_1 = np.sqrt(p_2_data * p_3_data / p_1_data)
    a_2 = np.sqrt(p_1_data * p_3_data / p_2_data)
    a_3 = np.sqrt(p_1_data * p_2_data / p_3_data)
    mu_1_data = np.abs(p_1_data) * np.sqrt(length_gap) / V_data
    mu_2_data = np.abs(p_2_data) * np.sqrt(length_gap) / V_data
    mu_3_data = np.abs(p_3_data) * np.sqrt(length_gap) / V_data
    #note: length_gap is constant and also this is a rewqritten version of the other mu_i definition(s) above. they are the same
    #holonomy variables
    b_1_data = mu_1_data * c_1_data
    b_2_data = mu_2_data * c_2_data
    b_3_data = mu_3_data * c_3_data
    f_1 = (np.sin(b_2_data) + np.sin(b_3_data)) * np.cos(b_1_data)
    f_2 = (np.sin(b_1_data) + np.sin(b_3_data)) * np.cos(b_2_data)
    f_3 = (np.sin(b_1_data) + np.sin(b_2_data)) * np.cos(b_3_data)
    # H is directional hubble parameter
    H_1 = (f_2 + f_3 - f_1) / (2 * gamma * length_gap)
    H_2 = (f_1 + f_3 - f_2) / (2 * gamma * length_gap)
    H_3 = (f_1 + f_2 - f_3) / (2 * gamma * length_gap)
    avg_H = (H_1 + H_2 + H_3) / 3
    rho = p_phi**2 / (2 * V_data**2)
    S_data = S_data = (np.sin(b_1_data) * np.sin(b_2_data) + np.sin(b_2_data) * np.sin(b_3_data) + np.sin(b_3_data) * np.sin(b_1_data))
    C_grav_data = (-V_data * S_data/ (8 * np.pi * G * gamma**2 * length_gap))
    C_matter_data = p_phi**2 / (2 * V_data)
    C_total = C_grav_data + C_matter_data
    """print("init constraint:", C_total[0])
    print("last constraint:", C_total[-1])
    print(" absolute max constraint:", np.max(np.abs(C_total)))"""
    results.append({"time":solution.t,"a_1": a_1, "a_2": a_2, "a_3": a_3, "H_1": H_1, "H_2": H_2, "H_3": H_3, "avg_H": avg_H, "volume": V_data, "rho": rho, "init_a": (init_a_x, init_a_y, init_a_z)})
#plots
colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

#a_i
fig1, ax1 = plt.subplots(figsize=(11, 7))
for i, result in enumerate(results):
    ax1.plot(result["time"], result["a_1"],
        color=colors[i], linestyle="-", alpha=0.6)
    ax1.plot(result["time"], result["a_2"],
        color=colors[i], linestyle="--", alpha=0.6)
    ax1.plot(result["time"], result["a_3"],
        color=colors[i], linestyle=":", alpha=0.6)
plt.figtext(0.15, 0.01, "a_1 is straight, a_2 is dashed, a_3 is dotted", ha="left", fontsize = 9, style = "normal", color = "black")
ax1.set_title(f"a_i against time for {n**3} trials")
ax1.set_xlabel("t")
ax1.set_ylabel("a_i")
ax1.grid()
plt.savefig('scale.pdf', format='pdf', bbox_inches='tight')

#H_i
fig2, ax2 = plt.subplots(figsize=(11, 7))
for i, result in enumerate(results):
    ax2.plot(result["time"], result["H_1"],
        color=colors[i], linestyle="-", alpha=0.6)
    ax2.plot(result["time"], result["H_2"],
        color=colors[i], linestyle="--", alpha=0.6)
    ax2.plot(result["time"], result["H_3"],
        color=colors[i], linestyle=":", alpha=0.6)
    ax2.plot(result["time"], result["avg_H"],
        color=colors[i], linestyle="-.", alpha=0.6)
plt.figtext(0.15, 0.01, "H_1 is straight, H_2 is dashed, H_3 is dotted, avg_H is dash-dotted", ha="left", fontsize = 9, style = "normal", color = "black")
ax2.set_title(f"H_i against time for {n**3} trials")
ax2.set_xlabel("t")
ax2.set_ylabel("H_i")
ax2.grid()
plt.savefig('hubble.pdf', format='pdf', bbox_inches='tight')

#V
fig3, ax3 = plt.subplots(figsize=(11, 7))
for i, result in enumerate(results):
    ax3.plot(result["time"], result["volume"],
        color=colors[i], alpha=0.7)

ax3.set_title(f"V against time for {n**3} trials")
ax3.set_xlabel("t")
ax3.set_ylabel("V")
ax3.grid()
plt.savefig('volume.pdf', format='pdf', bbox_inches='tight')

#rho
fig4, ax4 = plt.subplots(figsize=(11, 7))
for i, result in enumerate(results):
    ax4.plot(result["time"], result["rho"],
        color=colors[i], alpha=0.7)

ax4.set_title(f"rho against time for {n**3} trials")
ax4.set_xlabel("t")
ax4.set_ylabel("rho")
ax4.grid()

plt.savefig('rho.pdf', format='pdf', bbox_inches='tight')
print("done")
plt.show()

