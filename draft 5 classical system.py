import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import matplotlib as mpl
import sympy as sp
G = 1
rho_crit = 0.41
def lqc_dynamic_system(t, state):

    a_x = state[0]
    a_y = state[1]
    a_z = state[2]
    Hx = state[3] 
    Hy = state[4]
    Hz = state[5]
    if not np.all(np.isfinite(state)):
        print("Non-finite state encountered")

    if state[0] <= 0 or state[1] <= 0 or state[2] <= 0:
        print("Scale factor became non-positive")
    
    p_phi = 15.0 
    V = a_x * a_y * a_z

    rho = (p_phi**2) / (2.0 * V**2)

    dax_dt = a_x * Hx
    day_dt = a_y * Hy
    daz_dt = a_z * Hz
    avg_H = np.average([Hx, Hy, Hz])
    sigma_x = Hx - avg_H
    sigma_y = Hy - avg_H
    sigma_z = Hz - avg_H

    dHx_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_x
    dHy_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_y
    dHz_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_z
    return [dax_dt, day_dt, daz_dt, dHx_dt, dHy_dt, dHz_dt]

n = 5
x_scale = np.linspace(11, 90, n)
y_scale = np.linspace(11, 120, n)
z_scale = np.linspace(11, 99, n)

p_phi_val = 15 
t=1000000
time_span = (0, t) 
t_eval = np.linspace(time_span[0], time_span[1], t)

results = []
min_V = np.min(x_scale) * np.min(y_scale) * np.min(z_scale)
max_V = np.max(x_scale) * np.max(y_scale) * np.max(z_scale)
cmap = plt.get_cmap('plasma')

for initial_a_x, initial_a_y, initial_a_z in itertools.product(x_scale, y_scale, z_scale):
    V = initial_a_x * initial_a_y * initial_a_z
    initial_rho = (p_phi_val**2) / (2.0 * V**2)    
    initial_H_x = -11
    initial_H_y = -2
    initial_H_z = -13
    if initial_H_x * initial_H_y + initial_H_z * (initial_H_y + initial_H_x) == 8 * np.pi * G * initial_rho:
        print("satisfies classical system")
    else:
        break
    initial_state = [initial_a_x, initial_a_y, initial_a_z, initial_H_x, initial_H_y, initial_H_z]
    solution = solve_ivp(lqc_dynamic_system, time_span, initial_state, t_eval=t_eval, method="DOP853")
    t_steps = solution.t
    ax_data = solution.y[0]
    ay_data = solution.y[1]
    az_data = solution.y[2]
    Hx_data = solution.y[3]
    Hy_data = solution.y[4]
    Hz_data = solution.y[5]
    rho_data= (p_phi_val**2) /(2.0 * (ax_data * ay_data*az_data)**2)
    results.append({'params': (initial_a_x, initial_a_y, initial_a_z),
        't': t_steps,'ax': ax_data, 'ay': ay_data, 'az': az_data, 'Hx': Hx_data, 'Hy': Hy_data, 'Hz': Hz_data, 'rho': rho_data})


fig, ax = plt.subplots()
line_map = {}

for res in results:
    init_ax, init_ay, init_az = res["params"]
    V_init = float(init_ax * init_ay * init_az)
    color = cmap(np.clip((V_init - min_V) / (max_V - min_V), 0, 1))
    (line,) = ax.plot(res["t"], res["rho"], color=color, alpha=0.7, linewidth=1.5, picker=True, pickradius=5)
    line_map[line] = (init_ax, init_ay, init_az)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_V, vmax=max_V))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Initial Spatial Volume $V_0 = (a_x  \\times a_y \\times a_z)$",fontsize=11)

ax.set_title("Evolution of $\\rho(t)$ Across Anisotropic Grid",fontsize=12,fontweight="bold")
ax.set_xlabel("Cosmic Time ($t$)", fontsize=11)
ax.set_ylabel("Energy Density ($\\rho$)", fontsize=11)
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)

def on_pick(event):
    clicked_line = event.artist
    if clicked_line in line_map:
        ax_val, ay_val, az_val = line_map[clicked_line]
        print(f"Picked line initial conditions: ax={ax_val}, ay={ay_val}, az={az_val} @{time_span}")
fig.canvas.mpl_connect("pick_event", on_pick)

fig.tight_layout()
plt.show()
