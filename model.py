import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===========================
# Model Parameters
# ===========================

# Simulation parameters
simulation_time = 2.5  # seconds
dt = 0.001  # time step in seconds
time_points = np.arange(-0.5, simulation_time, dt)  # from -0.5s to 2.0s

# Network parameters
num_rows = 3  # Number of SAC rows
num_cols = 4  # Number of SAC columns
num_total_sacs = num_rows * num_cols  # Total number of SACs
dendrite_length = 0.1  # mm
cell_spacing = dendrite_length / 2  # mm, half dendritic length apart

# SAC structure
num_dendrites = 6  # Number of dendrites per SAC
compartments_per_dendrite = 2  # proximal and distal
num_total_dendritic_compartments = num_total_sacs * num_dendrites * compartments_per_dendrite  # 3*4*6*2=144
num_soma_compartments = num_total_sacs  # one soma per SAC
num_total_compartments = num_total_dendritic_compartments + num_soma_compartments  # 153

# Conductance and reversal potentials (in mV and mS/cm^2)
E_glu = 0.0  # Glutamate reversal potential
E_K = -80.0  # K+ reversal potential
E_Cl_proximal = -50.0  # Cl- reversal potential proximal
E_Cl_distal = -80.0  # Cl- reversal potential distal

g_glu_rest = 0.0  # Resting glutamate conductance
g_glu_bound = 1.0  # Bound glutamate conductance (high value)
g_K = 0.2  # K+ conductance
g_Cl_rest = 0.0  # Resting Cl- conductance
g_Cl_bound = 1.0  # Bound Cl- conductance (high value)

# GABA kinetics
alpha = 240.0  # Activation rate (s^-1)
beta = 18.0    # Deactivation rate (s^-1)
theta1 = -50.0  # Threshold for s1 activation (mV)
theta2 = 0.6    # Threshold for s2 activation (dimensionless)

# Coupling between compartments (mS/cm^2)
delta = 0.5

# Membrane time constant (seconds)
tau = 1.0

# ===========================
# Helper Functions
# ===========================

def H1(v):
    """Sigmoidal function for H1(v) centered at theta1."""
    k = 1.0  # Slope parameter
    return 1.0 / (1.0 + np.exp(-k * (v - theta1)))

def H2(s1):
    """Sigmoidal function for H2(s1) centered at theta2."""
    k = 1.0  # Slope parameter
    return 1.0 / (1.0 + np.exp(-k * (s1 - theta2)))

# ===========================
# Network Initialization
# ===========================

class Compartment:
    """Represents a compartment in a SAC."""
    def __init__(self, sac_id, dendrite_id, compartment_id):
        self.sac_id = sac_id
        self.dendrite_id = dendrite_id
        self.compartment_id = compartment_id  # 0: proximal, 1: distal
        self.id = f"SAC{self.sac_id}_D{self.dendrite_id}_C{self.compartment_id}"
        self.E_Cl = E_Cl_proximal if compartment_id == 0 else E_Cl_distal
        self.v = -70.0  # Initial voltage (mV)
        self.s1 = 0.0  # GABA gating variable s1
        self.s2 = 0.0  # GABA gating variable s2

class SAC:
    """Represents a Starburst Amacrine Cell (SAC)."""
    def __init__(self, sac_id, row, col):
        self.sac_id = sac_id
        self.row = row
        self.col = col
        self.dendrites = [
            Compartment(sac_id, d, c) 
            for d in range(num_dendrites) 
            for c in range(compartments_per_dendrite)
        ]
        self.soma = Compartment(sac_id, 'soma', 'soma')  # Soma compartment

# Create the network of SACs
network = []
for sac_id in range(num_total_sacs):
    row = sac_id // num_cols
    col = sac_id % num_cols
    network.append(SAC(sac_id, row, col))

# ===========================
# Glutamate Stimulus Function
# ===========================

def glutamate_stimulus(t, sac, dendrite, compartment):
    """
    Determine the glutamate conductance at time t for a given compartment.
    Simulates a moving glutamate wave from left to right.
    """
    # Parameters for the glutamate wave
    signal_speed = 0.5  # mm/s
    signal_width = 0.2  # mm
    signal_position = signal_speed * t  # Current position of the wave center
    
    # Spatial position of the compartment
    x_base = sac.col * cell_spacing  # Base x-position based on column
    angle = dendrite * 60.0  # Degrees, assuming hexagonal symmetry
    x_pos = x_base + dendrite_length * np.cos(np.deg2rad(angle))
    
    # Check if the compartment is within the glutamate wave
    if (signal_position - signal_width / 2) <= x_pos <= (signal_position + signal_width / 2):
        return g_glu_bound
    else:
        return g_glu_rest

# ===========================
# ODE System Definition
# ===========================

def model(t, y, network, num_sacs, num_dendrites, compartments_per_dendrite, g_glu_bound, g_glu_rest):
    """
    Defines the ODEs for the SAC network.
    """
    dy = np.zeros_like(y)
    
    # Extract voltages and gating variables from state vector
    V = y[:num_total_compartments]  # Voltages
    s1 = y[num_total_compartments:2*num_total_compartments]  # GABA gating variable s1
    s2 = y[2*num_total_compartments:3*num_total_compartments]  # GABA gating variable s2
    
    # Iterate over all SACs
    for sac in network:
        # Iterate over all dendrites and compartments
        for dendrite in range(num_dendrites):
            for compartment in range(compartments_per_dendrite):
                # Calculate the index in the state vector
                compartment_global_index = sac.sac_id * num_dendrites * compartments_per_dendrite + dendrite * compartments_per_dendrite + compartment
                # Get compartment object
                comp = sac.dendrites[dendrite * compartments_per_dendrite + compartment]
                # Current voltage
                v = V[compartment_global_index]
                # Current gating variables
                sc1 = s1[compartment_global_index]
                sc2 = s2[compartment_global_index]
                
                # Determine glutamate conductance based on stimulus
                g_glu = glutamate_stimulus(t, sac, dendrite, compartment)
                
                # Calculate H1 and H2
                H1_val = H1(v)
                H2_val = H2(sc1)
                
                # Update gating variables
                dy[num_total_compartments + compartment_global_index] = alpha * (1.0 - sc1) * H1_val - beta * sc1
                dy[2*num_total_compartments + compartment_global_index] = alpha * (1.0 - sc2) * H2_val - beta * sc2
                
                # Calculate GABAergic conductance
                # Sum s2 from all distal compartments co-localized with this compartment (same SAC)
                g_Cl = g_Cl_rest
                for other_dendrite in range(num_dendrites):
                    # Only distal compartments contribute
                    other_compartment = other_dendrite * compartments_per_dendrite + 1  # distal
                    other_global_index = sac.sac_id * num_dendrites * compartments_per_dendrite + other_dendrite * compartments_per_dendrite + other_compartment
                    g_Cl += (g_Cl_bound - g_Cl_rest) * s2[other_global_index]
                
                # Update voltage
                dvdt = (1.0 / tau) * (comp.E_Cl - v) \
                       + g_glu * (E_glu - v) \
                       + g_K * (E_K - v) \
                       + g_Cl * (comp.E_Cl - v)
                
                # Add coupling to adjacent compartments within the same dendrite
                if compartment == 0:
                    # Proximal compartment connected to distal
                    distal_compartment_global_index = compartment_global_index + 1
                    if distal_compartment_global_index < num_total_compartments:
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
    
        # Update soma compartment
        soma_global_index = num_total_dendritic_compartments + sac.sac_id  # Soma indices start after dendritic compartments
        # Soma voltage
        v_soma = V[soma_global_index]
        # Soma has only K+ channels and coupling to all proximal dendrites
        dvdt_soma = (1.0 / tau) * (E_K - v_soma)
        for dendrite in range(num_dendrites):
            proximal_compartment = dendrite * compartments_per_dendrite + 0  # proximal
            proximal_global_index = sac.sac_id * num_dendrites * compartments_per_dendrite + proximal_compartment
            proximal_v = V[proximal_global_index]
            dvdt_soma += delta * (proximal_v - v_soma)
        # Assign dvdt to dy
        dy[soma_global_index] = dvdt_soma
    
    return dy

# ===========================
# State Vector Initialization
# ===========================

# Total state vector size: [V (153)] + [s1 (153)] + [s2 (153)] = 459
total_state_size = num_total_compartments * 3

# Initialize state vector
y0 = np.full(total_state_size, -70.0)  # Initialize all voltages to -70 mV

# Initialize soma compartments to -70 mV (redundant, but explicit)
for sac in network:
    soma_index_in_V = num_total_dendritic_compartments + sac.sac_id
    y0[soma_index_in_V] = -70.0

# ===========================
# Run the Simulation
# ===========================

def run_simulation():
    """Runs the ODE simulation."""
    # Define the ODE function with updated network and parameters
    def ode_system(t, y):
        return model(t, y, network, num_total_sacs, num_dendrites, compartments_per_dendrite, g_glu_bound, g_glu_rest)
    
    print("Starting simulation...")
    sol = solve_ivp(ode_system, [time_points[0], time_points[-1]], y0, t_eval=time_points, method='RK45', vectorized=False)
    print("Simulation completed.")
    
    return sol

# ===========================
# Extract and Plot Results
# ===========================

def plot_results(sol):
    """
    Extracts voltages from soma and dendritic tips and plots them.
    """
    # Extract voltages and gating variables from solution
    V = sol.y[:num_total_compartments, :]  # Voltages
    s1 = sol.y[num_total_compartments:2*num_total_compartments, :]  # GABA gating variable s1
    s2 = sol.y[2*num_total_compartments:3*num_total_compartments, :]  # GABA gating variable s2
    
    # Select a specific SAC to analyze, e.g., the center SAC
    # For even grid sizes, select the first central SAC
    center_sac_index = num_total_sacs // 2  # For 12 SACs, index 6 (0-based)
    center_sac = network[center_sac_index]
    sac_id = center_sac.sac_id
    
    # Correct soma index calculation
    soma_index_in_V = num_total_dendritic_compartments + sac_id  # Soma index in V
    if soma_index_in_V >= num_total_compartments:
        raise ValueError(f"Soma index {soma_index_in_V} out of bounds for V with {num_total_compartments} compartments.")
    
    center_soma_voltage = V[soma_index_in_V, :]
    
    # Select dendritic tips: distal compartments
    # For the center SAC, collect all distal compartments
    center_dendritic_tips_indices = [
        sac_id * num_dendrites * compartments_per_dendrite + dendrite * compartments_per_dendrite + 1
        for dendrite in range(num_dendrites)
    ]
    
    # Extract voltages for center dendritic tips
    center_dendritic_tips_voltage = V[center_dendritic_tips_indices, :]
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(sol.t, center_soma_voltage, label='Center Soma Voltage', linewidth=2)
    for i, dendritic_voltage in enumerate(center_dendritic_tips_voltage):
        plt.plot(sol.t, dendritic_voltage, label=f'Center Dendrite {i+1} Voltage', alpha=0.6)
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Soma and Dendritic Tip Voltages Over Time for Center SAC')
    plt.legend()
    plt.grid(True)
    plt.show()

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Run the simulation
    sol = run_simulation()
    
    # Plot the results
    plot_results(sol)
