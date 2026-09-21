import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import matplotlib as mpl
G = 1
rho_crit = 1.0  # Critical maximum density threshold in QLC
def lqc_dynamic_system(t, state):
    #state vector
    #state[0, 1, 2] are for each separate scaling factor
    #state[3, 4, 5] is for hubble parameter
    a_x = state[0]
    a_y = state[1]
    a_z = state[2]
    Hx = state[3] #its meant to by H_x, H_y, etc but already used _ as sub for / tl;dr Hx means H_x not H*x
    Hy = state[4]
    Hz = state[5]
    if np.isclose(a_x, 0, atol=1e-5) or np.isclose(a_y, 0, atol=1e-5) or np.isclose(a_z, 0, atol=1e-5):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #conservation of momentum, must be >1
    p_phi = 15.0 
    V = a_x * a_y * a_z
    # Energy density of scalar field: \rho = p_\phi^2 / (2 * a^6)
    rho = (p_phi**2) / (2.0 * V**2)
    
    #scale factor derivative: \dot{a} = a * H, from Friedmann
    dax_dt = a_x * Hx
    day_dt = a_y * Hy
    daz_dt = a_z * Hz
    avg_H = np.average([Hx, Hy, Hz])
    sigma_x = Hx - avg_H
    sigma_y = Hy - avg_H
    sigma_z = Hz - avg_H
    # raychaudi equation
    dHx_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_x
    dHy_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_y
    dHz_dt = -8.0 * np.pi * G * rho * (1.0 - (rho / rho_crit)) - 3 * avg_H * sigma_z
    return [dax_dt, day_dt, daz_dt, dHx_dt, dHy_dt, dHz_dt]
#any numbers here, make sure last numbers are the same. 
#keep n > 0, also keep it an integer, also computer will do n**3 calcs so dont set above 100
#x_scale[1]~y_scale[1]~z_scale[1]
n = 3
x_scale = np.linspace(11, 98, n)
y_scale = np.linspace(11, 99, n)
z_scale = np.linspace(11, 100, n)

p_phi_val = 15  # same as p_phi
#would reccomend t >> 2nd number 
t=500
time_span = (0, t)  #next time step
#causes infty error past ~100,000, if you dont seee the bounce jsut increase this number
#in t_eval can increase last number to a multiple of t (e.g. 100t) to create more detailed graph. 
#for now its 1t which is equal to like one per time step but it can be more detailed to approximate continuous universes. mine is lqc so i will stick with 1t
t_eval = np.linspace(time_span[0], time_span[1], t)

results = []
min_V = np.min(x_scale) * np.min(y_scale) * np.min(z_scale)
max_V = np.max(x_scale) * np.max(y_scale) * np.max(z_scale)
cmap = plt.get_cmap('plasma')

for initial_a_x, initial_a_y, initial_a_z in itertools.product(x_scale, y_scale, z_scale):
    V = initial_a_x * initial_a_y * initial_a_z
    initial_rho = (p_phi_val**2) / (2.0 * V**2)

    #start h as negative
    initial_H = -np.sqrt((8.0 * np.pi * G / 3.0)* initial_rho* (1.0 - (initial_rho / rho_crit))  )
    initial_state = [initial_a_x, initial_a_y, initial_a_z, initial_H, initial_H, initial_H]

    #can change integration method, rk45 is default but for stiff equations 
    solution = solve_ivp(
          lqc_dynamic_system, time_span, initial_state, t_eval=t_eval, method='BDF'
    )

    #extract solutions of each equation
    t_steps = solution.t
    ax_data = solution.y[0]
    ay_data = solution.y[1]
    az_data = solution.y[2]
    Hx_data = solution.y[3]
    Hy_data = solution.y[4]
    Hz_data = solution.y[5]
    rho_data = (p_phi_val**2) / (2.0 * (ax_data * ay_data*az_data)**2)
    results.append(
        {'params': (initial_a_x, initial_a_y, initial_a_z),
          't': t_steps,'ax': ax_data, 'ay': ay_data, 'az': az_data, 'Hx': Hx_data, 'Hy': Hy_data, 'Hz': Hz_data, 'rho': rho_data}

    )


fig, ax = plt.subplots()

line_map = {}

for res in results:
    init_ax, init_ay, init_az = res["params"]
    V_init = float(init_ax * init_ay * init_az)

    # map volume to colour
    color = cmap(np.clip((V_init - min_V) / (max_V - min_V), 0, 1))

    # 1. Enable picker=True directly on each line plot (not the grid)
    # pickradius defines how close (in points) the mouse needs to click to hit the line
    (line,) = ax.plot(res["t"], res["rho"], color=color, alpha=0.7, linewidth=1.5, picker=True, pickradius=5)

    # 2. Map the generated line object to its initial conditions
    line_map[line] = (init_ax, init_ay, init_az)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_V, vmax=max_V))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Initial Spatial Volume $V_0 = (a_x  \\times a_y \\times a_z)$",fontsize=11)

ax.set_title("Evolution of Energy Density $\\rho(t)$ Across Anisotropic Grid",fontsize=12,fontweight="bold")
ax.set_xlabel("Cosmic Time ($t$)", fontsize=11)
ax.set_ylabel("Energy Density ($\\rho$)", fontsize=11)
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)

#picking callback function
def on_pick(event):
    # event.artist contains the specific Line2D object that was clicked
    clicked_line = event.artist

    if clicked_line in line_map:
        ax_val, ay_val, az_val = line_map[clicked_line]
        print(f"Picked line initial conditions: ax={ax_val}, ay={ay_val}, az={az_val} @{time_span}")


# 4. Connect the canvas pick event to your callback
fig.canvas.mpl_connect("pick_event", on_pick)

fig.tight_layout()
plt.show()

