# model.py

import numpy as np

# ===========================
# Helper Functions
# ===========================

def H1(v, theta1=-50.0, k=1.0):
    """
    Sigmoidal function for H1(v) centered at theta1.

    Parameters:
    - v: Membrane potential (mV)
    - theta1: Threshold potential for activation (mV)
    - k: Slope parameter

    Returns:
    - Sigmoid activation value between 0 and 1
    """
    x = k * (v - theta1)
    # Clip x to prevent overflow in exp
    x = np.clip(x, -700, 700)  # np.exp(-700) is effectively 0, np.exp(700) is large but manageable
    return 1.0 / (1.0 + np.exp(-x))

def H2(s1, theta2=0.6, k=1.0):
    """
    Sigmoidal function for H2(s1) centered at theta2.

    Parameters:
    - s1: GABA gating variable s1 (dimensionless)
    - theta2: Threshold for s2 activation (dimensionless)
    - k: Slope parameter

    Returns:
    - Sigmoid activation value between 0 and 1
    """
    x = k * (s1 - theta2)
    # Clip x to prevent overflow in exp
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))

# ===========================
# Biological Components
# ===========================

class Compartment:
    """
    Represents a compartment within a Starburst Amacrine Cell (SAC).

    Attributes:
    - sac_id: Identifier for the SAC
    - dendrite_id: Identifier for the dendrite within the SAC
    - compartment_id: 0 for proximal, 1 for distal, 'soma' for soma
    - id: Unique string identifier
    - E_Cl: Chloride reversal potential (mV)
    - v: Membrane potential (mV)
    - s1: GABA gating variable s1 (dimensionless)
    - s2: GABA gating variable s2 (dimensionless)
    """
    def __init__(self, sac_id, dendrite_id, compartment_id, E_Cl_proximal=-50.0, E_Cl_distal=-80.0, E_Cl_soma=-80.0):
        self.sac_id = sac_id
        self.dendrite_id = dendrite_id
        self.compartment_id = compartment_id  # 0: proximal, 1: distal, 'soma' for soma
        self.id = f"SAC{self.sac_id}_D{self.dendrite_id}_C{self.compartment_id}"
        if compartment_id == 'soma':
            self.E_Cl = E_Cl_soma
        elif compartment_id == 0:
            self.E_Cl = E_Cl_proximal
        elif compartment_id == 1:
            self.E_Cl = E_Cl_distal
        else:
            raise ValueError("Invalid compartment_id: must be 0 (proximal), 1 (distal), or 'soma'")
        self.v = -70.0  # mV, resting potential
        self.s1 = 0.0  # GABA gating variable s1
        self.s2 = 0.0  # GABA gating variable s2

class SAC:
    """
    Represents a Starburst Amacrine Cell (SAC).

    Attributes:
    - sac_id: Identifier for the SAC
    - row: Row position in the network grid
    - col: Column position in the network grid
    - dendrites: List of Compartment objects representing dendrites
    - soma: Compartment object representing the soma
    """
    def __init__(self, sac_id, row, col, num_dendrites=6, compartments_per_dendrite=2):
        self.sac_id = sac_id
        self.row = row
        self.col = col
        self.dendrites = [
            Compartment(sac_id, d, c) 
            for d in range(num_dendrites) 
            for c in range(compartments_per_dendrite)
        ]
        self.soma = Compartment(sac_id, 'soma', 'soma')  # Soma compartment

class Network:
    """
    Represents the entire network of Starburst Amacrine Cells (SACs).

    Attributes:
    - num_rows: Number of SAC rows in the grid
    - num_cols: Number of SAC columns in the grid
    - num_dendrites: Number of dendrites per SAC
    - compartments_per_dendrite: Number of compartments per dendrite
    - sacs: List of SAC objects in the network
    - num_total_sacs: Total number of SACs
    - num_total_dendritic_compartments: Total number of dendritic compartments across all SACs
    - num_soma_compartments: Total number of soma compartments
    - num_total_compartments: Total number of compartments (dendritic + soma)
    """
    def __init__(self, num_rows=3, num_cols=4, num_dendrites=6, compartments_per_dendrite=2):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_dendrites = num_dendrites
        self.compartments_per_dendrite = compartments_per_dendrite
        self.sacs = []
        sac_id = 0
        for row in range(num_rows):
            for col in range(num_cols):
                self.sacs.append(SAC(sac_id, row, col, num_dendrites, compartments_per_dendrite))
                sac_id += 1
        self.num_total_sacs = sac_id
        self.num_total_dendritic_compartments = self.num_total_sacs * self.num_dendrites * self.compartments_per_dendrite
        self.num_soma_compartments = self.num_total_sacs
        self.num_total_compartments = self.num_total_dendritic_compartments + self.num_soma_compartments
