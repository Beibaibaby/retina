# simulation.py

import numpy as np
from scipy.integrate import solve_ivp
from model import Network, H1, H2

# ===========================
# Global Model Parameters
# ===========================

# Conductance and reversal potentials (consistent with model.py)
E_glu = 0.0  # mV
E_K = -80.0  # mV
E_Cl_proximal = -50.0  # mV
E_Cl_distal = -80.0  # mV
E_Cl_soma = -80.0  # mV

g_glu_rest = 0.0  # mS/cm^2
g_glu_bound = 1.0  # mS/cm^2 (arbitrary large value)
g_K = 0.2  # mS/cm^2
g_Cl_rest = 0.0  # mS/cm^2
g_Cl_bound = 1.0  # mS/cm^2 (arbitrary large value)

# GABA kinetics
alpha = 240.0  # s^-1
beta = 18.0    # s^-1
theta1 = -50.0  # mV
theta2 = 0.6    # dimensionless

# Coupling between compartments (mS/cm^2)
delta = 0.5

# Membrane time constant (seconds)
tau = 1.0

# ===========================
# Stimulus Function
# ===========================

def glutamate_stimulus(t, sac, dendrite, compartment, cell_spacing=0.05, dendrite_length=0.1):
    """
    Determine the glutamate conductance at time t for a given compartment.
    Simulates a moving glutamate wave from left to right.

    Parameters:
    - t: Current time (s)
    - sac: SAC object
    - dendrite: Dendrite index
    - compartment: Compartment index within the dendrite
    - cell_spacing: Spatial spacing between SACs (mm)
    - dendrite_length: Length of each dendrite (mm)

    Returns:
    - Glutamate conductance value (mS/cm^2)
    """
    # Parameters for the glutamate wave
    signal_speed = 0.5  # mm/s
    signal_width = 0.2  # mm
    signal_position = signal_speed * t  # Current position of the wave center

    # Spatial position of the compartment
    x_base = sac.col * cell_spacing  # base x-position based on column
    angle = dendrite * 60.0  # degrees
    x_pos = x_base + dendrite_length * np.cos(np.deg2rad(angle))

    # Check if the compartment is within the glutamate wave
    if (signal_position - signal_width / 2) <= x_pos <= (signal_position + signal_width / 2):
        return g_glu_bound
    else:
        return g_glu_rest

# ===========================
# ODE System Definition
# ===========================

def odes(t, y, network):
    """
    Defines the ODEs for the SAC network.

    Parameters:
    - t: Current time (s)
    - y: State vector containing [V, s1, s2] for all compartments
    - network: Network object

    Returns:
    - dy/dt: Derivative of the state vector
    """
    dy = np.zeros_like(y)

    # Extract voltages and gating variables from state vector
    V = y[:network.num_total_compartments]  # Voltages
    s1 = y[network.num_total_compartments:2*network.num_total_compartments]  # GABA gating variable s1
    s2 = y[2*network.num_total_compartments:3*network.num_total_compartments]  # GABA gating variable s2

    # Iterate over all SACs
    for sac in network.sacs:
        # Iterate over all dendrites and compartments
        for dendrite in range(network.num_dendrites):
            for compartment in range(network.compartments_per_dendrite):
                # Calculate the index in the state vector
                compartment_global_index = sac.sac_id * network.num_dendrites * network.compartments_per_dendrite + dendrite * network.compartments_per_dendrite + compartment
                # Get compartment object
                comp = sac.dendrites[dendrite * network.compartments_per_dendrite + compartment]
                # Current voltage
                v = V[compartment_global_index]
                # Current gating variables
                sc1 = s1[compartment_global_index]
                sc2 = s2[compartment_global_index]

                # Determine glutamate conductance based on stimulus
                g_glu = glutamate_stimulus(t, sac, dendrite, compartment)

                # Calculate H1 and H2
                H1_val = H1(v, theta1=theta1, k=1.0)
                H2_val = H2(sc1, theta2=theta2, k=1.0)

                # Update gating variables
                dy[network.num_total_compartments + compartment_global_index] = alpha * (1.0 - sc1) * H1_val - beta * sc1
                dy[2*network.num_total_compartments + compartment_global_index] = alpha * (1.0 - sc2) * H2_val - beta * sc2

                # Calculate GABAergic conductance
                # Sum s2 from all distal compartments co-localized with this compartment (same SAC)
                g_Cl = g_Cl_rest
                for other_dendrite in range(network.num_dendrites):
                    other_compartment = other_dendrite * network.compartments_per_dendrite + 1  # distal
                    other_global_index = sac.sac_id * network.num_dendrites * network.compartments_per_dendrite + other_dendrite * network.compartments_per_dendrite + other_compartment
                    if other_global_index >= network.num_total_compartments:
                        print(f"Warning: other_global_index {other_global_index} out of bounds for s2 with size {len(s2)}")
                        continue
                    g_Cl += (g_Cl_bound - g_Cl_rest) * s2[other_global_index]

                # Update voltage
                dvdt = (1.0 / tau) * (comp.E_Cl - v) + g_glu * (E_glu - v) + g_K * (E_K - v) + g_Cl * (comp.E_Cl - v)

                # Add coupling to adjacent compartments within the same dendrite
                if compartment == 0:
                    # Proximal compartment connected to distal
                    distal_compartment_global_index = compartment_global_index + 1
                    if distal_compartment_global_index < network.num_total_compartments:
                        distal_v = V[distal_compartment_global_index]
                        dvdt += delta * (distal_v - v)
                elif compartment == 1:
                    # Distal compartment connected to proximal
                    proximal_compartment_global_index = compartment_global_index - 1
                    if proximal_compartment_global_index >= 0:
                        proximal_v = V[proximal_compartment_global_index]
                        dvdt += delta * (proximal_v - v)

                # Assign dvdt to dy
                dy[compartment_global_index] = dvdt

    # Iterate over all SACs to update soma compartments
    for sac in network.sacs:
        # Soma compartment index
        soma_global_index = network.num_total_dendritic_compartments + sac.sac_id
        # Soma voltage
        v_soma = V[soma_global_index]
        # Soma has only K+ channels and coupling to all proximal dendrites
        dvdt_soma = (1.0 / tau) * (E_K - v_soma)
        for dendrite in range(network.num_dendrites):
            proximal_compartment = dendrite * network.compartments_per_dendrite + 0  # proximal
            proximal_global_index = sac.sac_id * network.num_dendrites * network.compartments_per_dendrite + proximal_compartment
            proximal_v = V[proximal_global_index]
            dvdt_soma += delta * (proximal_v - v_soma)
        # Assign dvdt to dy
        dy[soma_global_index] = dvdt_soma

    return dy

# ===========================
# Simulation Function
# ===========================

def run_simulation(network, time_points):
    """
    Runs the ODE simulation for the given network and time points.

    Parameters:
    - network: Network object
    - time_points: Array of time points to evaluate (s)

    Returns:
    - sol: Solution object from solve_ivp
    """
    # Initial state vector: [V (455), s1 (455), s2 (455)] = 1365
    y0 = np.full(network.num_total_compartments * 3, -70.0)  # Initialize all voltages to -70 mV

    # Initialize soma compartments to -70 mV explicitly
    for sac in network.sacs:
        soma_index_in_V = network.num_total_dendritic_compartments + sac.sac_id
        y0[soma_index_in_V] = -70.0  # Already set, but explicit

    # Define the ODE function with updated network and parameters
    def ode_func(t, y):
        return odes(t, y, network)

    # Run the solver
    print("Starting simulation...")
    sol = solve_ivp(ode_func, [time_points[0], time_points[-1]], y0, t_eval=time_points, method='RK45', vectorized=False)
    print("Simulation completed.")

    return sol
