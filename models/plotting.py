# plotting.py

import numpy as np
import matplotlib.pyplot as plt

def plot_voltage_traces(sol, network, time_points):
    """
    Plots the soma voltage and dendritic tip voltages of the center SAC.

    Parameters:
    - sol: Solution object from solve_ivp
    - network: Network object
    - time_points: Array of time points corresponding to sol.t
    """
    # Extract voltages from solution
    V = sol.y[:network.num_total_compartments, :]  # Voltages

    # Select a specific SAC to analyze, e.g., the center SAC
    center_sac_index = network.num_total_sacs // 2  # For 35 SACs, index 17 (0-based)
    center_sac = network.sacs[center_sac_index]
    sac_id = center_sac.sac_id

    # Soma index in V
    soma_index_in_V = network.num_total_dendritic_compartments + sac_id
    if soma_index_in_V >= network.num_total_compartments:
        raise ValueError(f"Soma index {soma_index_in_V} out of bounds for V with {network.num_total_compartments} compartments.")

    center_soma_voltage = V[soma_index_in_V, :]

    # Dendritic tip indices for the center SAC (distal compartments)
    center_dendritic_tips_indices = [
        sac_id * network.num_dendrites * network.compartments_per_dendrite + dendrite * network.compartments_per_dendrite + 1
        for dendrite in range(network.num_dendrites)
    ]

    # Extract voltages for center dendritic tips
    center_dendritic_tips_voltage = V[center_dendritic_tips_indices, :]

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(sol.t, center_soma_voltage, label='Center Soma Voltage', linewidth=2, color='blue')
    colors = plt.cm.viridis(np.linspace(0, 1, network.num_dendrites))
    for i, dendritic_voltage in enumerate(center_dendritic_tips_voltage):
        plt.plot(sol.t, dendritic_voltage, label=f'Center Dendrite {i+1} Voltage', alpha=0.6, color=colors[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Soma and Dendritic Tip Voltages Over Time for Center SAC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_parameter_sensitivity(sol, network, time_points, parameter='g_Cl_bound', values=[0.5, 1.0, 1.5]):
    """
    Plots parameter sensitivity by varying a specific parameter and observing changes in DS.

    Parameters:
    - sol: Solution object from solve_ivp (current simulation)
    - network: Network object
    - time_points: Array of time points
    - parameter: Parameter to vary (e.g., 'g_Cl_bound')
    - values: List of values to assign to the parameter
    """
    # Placeholder function for parameter sensitivity
    # To implement, you need to modify the network's parameters, rerun simulations, and plot results
    pass  # Implement as needed
