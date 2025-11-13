# ==============================================================================
"""
CONSTRUCTIVE HEURISTIC FILE

This file contains classes/functions to generate initial fleet capacity
solutions, which can then be fed into the main solver.
"""
# ==============================================================================

class ConstructiveHeuristic:
    
    def __init__(self):
        """
        Initialize the heuristic. (Can be expanded later with
        different strategies or parameters).
        """
        print(f"--- Heuristic: Initializing Constructive Heuristic ---")
        
    def generate_solution(self, G_set, S_set, T_minus_set):
        """
        Generates a fixed fleet solution.
        
        INPUTS:
        - G_set: The Pyomo RangeSet for vehicle groups (e.g., model.G)
        - S_set: The Pyomo RangeSet for stations (e.g., model.S)
        - T_minus_set: The Pyomo RangeSet for time (e.g., model.T_minus)
        
        OUTPUTS:
        - w_O_fixed (dict): A dictionary mapping (g, s) to a fixed integer value.
        - w_L_fixed (dict): A dictionary mapping (g, t, s) to a fixed integer value.
        """
        
        print("--- Heuristic: Generating simple 'buy/lease 1 of everything' solution... ---")
        
        # --- Example Heuristic: Buy/Lease 1 of Everything ---
        # As you requested, this simple version just sets all values to 1.
        
        w_O_fixed = {}
        for g in G_set:
            for s in S_set:
                w_O_fixed[(g, s)] = 1
                
        w_L_fixed = {}
        for g in G_set:
            for t in T_minus_set:
                for s in S_set:
                    w_L_fixed[(g, t, s)] = 1
        
        print(f"--- Heuristic: Generated {len(w_O_fixed)} w_O decisions and {len(w_L_fixed)} w_L decisions. ---")
        print(f"Size: w_O_fixed = {len(w_O_fixed)}, w_L_fixed = {len(w_L_fixed)}")
        
        return w_O_fixed, w_L_fixed

    def generate_zero_solution(self, G_set, S_set, T_minus_set):
        """
        Generates a "zero fleet" solution.
        """
        print("--- Heuristic: Generating 'zero fleet' solution... ---")
        w_O_fixed = {(g, s): 0 for g in G_set for s in S_set}
        w_L_fixed = {(g, t, s): 0 for g in G_set for t in T_minus_set for s in S_set}
        print(f"--- Heuristic: Generated {len(w_O_fixed)} w_O decisions and {len(w_L_fixed)} w_L decisions. ---")
        return w_O_fixed, w_L_fixed