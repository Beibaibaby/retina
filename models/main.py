# main.py
import numpy as np
from model import Network
from simulation import run_simulation
from plotting import plot_voltage_traces

def main():
    """
    Main function to set up the network, run the simulation, and plot results.
    """
    # Define simulation time
    simulation_time = 2.5  # seconds
    dt = 0.01  # time step in seconds
    time_points = np.arange(-0.5, simulation_time, dt)  # from -0.5s to 2.0s

    # Initialize the network (as per the paper's example: 5 rows x 7 columns)
    num_rows = 5  # Number of SAC rows
    num_cols = 7  # Number of SAC columns
    network = Network(num_rows=num_rows, num_cols=num_cols, num_dendrites=6, compartments_per_dendrite=2)

    # Run the simulation
    sol = run_simulation(network, time_points)

    # Plot the voltage traces
    plot_voltage_traces(sol, network, time_points)

    # Additional plotting functions can be called here
    # For example:
    # plot_parameter_sensitivity(sol, network, time_points, parameter='g_Cl_bound', values=[0.5, 1.0, 1.5])

if __name__ == "__main__":
    main()
