# ==============================================================================
"""
VALIDATION SCRIPT: FULL SOLUTION CHECKER

This script performs a two-step process to validate the linearization:

1.  SOLVE MILP: It solves the (corrected) linearized MILP to get the
    optimal solution for ALL decision variables (w, u, y, x, f, q).
2.  TEST MINLP: It builds the original non-linear MINLP, FIXES ALL
    variables to the solution from Step 1, and "solves".

This checks two things:
1.  Feasibility: Is the linear solution feasible in the non-linear model?
2.  Correctness: Does the linear objective value (using 'v') equal the
    non-linear objective value (using 'u' and 'q')?
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
from pyomo.environ import *
import pandas as pd
import numpy as np

# ==============================================================================
# MAIN VALIDATION FUNCTION
# ==============================================================================

def validate_full_solution(filename):
    print(f"\n{'='*80}")
    print(f"STARTING FULL SOLUTION VALIDATION FOR: {filename}")
    print(f"{'='*80}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return

    # ==========================================================================
    # --- STEP 1: LOAD DATA ---
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: LOADING DATA from {filename}")
    print(f"{'='*60}")
    
    try:
        xls = pd.ExcelFile(filename)

        # --- HELPER ---
        def get_clean_matrix(sheet_name):
            df = pd.read_excel(xls, sheet_name, header=None)
            df_num = df.apply(pd.to_numeric, errors='coerce')
            return df_num.dropna(how='all', axis=0).dropna(how='all', axis=1).values

        # --- 1. Unit Parameters ---
        up_df = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        up_df.index = up_df.index.astype(str).str.strip()
        G = int(up_df.loc['G'].iloc[0])
        S = int(up_df.loc['S'].iloc[0])
        A = int(up_df.loc['A'].iloc[0])
        T = int(up_df.loc['T'].iloc[0])
        PYU = float(up_df.loc['PYU'].iloc[0])
        BUD = float(up_df.loc['BUD'].iloc[0])

        # --- 2. Rental Types (Robust) ---
        rentals_df = pd.read_excel(xls, 'RentalTypes')
        r_pot = rentals_df.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot) == 0:
            r_pot = rentals_df.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        r_data = r_pot.values.astype(int)
        R = len(r_data)

        # --- 3. Parameters By Group ---
        pbg_df = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        pbg_df.index = pbg_df.index.astype(str).str.strip()
        def get_gp(name, dtype=float):
            vals = pbg_df.loc[name].values.flatten()
            vals = vals[:G] if len(vals) >= G else [vals[0]] * G
            return {g: dtype(vals[g]) for g in range(G)}
        LEA_g = get_gp('LEA_g')
        LP_g = get_gp('LP_g', int)
        COS_g = get_gp('COS_g')
        OWN_g = get_gp('OWN_g')

        # --- 4. Prices ---
        pri_data = get_clean_matrix('Prices')
        PRI_pg = {(p, g): pri_data[p, g] for p in range(pri_data.shape[0]) for g in range(G)}
        P = pri_data.shape[0]

        # --- 5. Demand ---
        dem_raw = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dem_nums = [x for x in dem_raw if isinstance(x, (int, float)) and not np.isnan(x)]
        req_size = R * (A + 1) * P
        if len(dem_nums) < req_size:
            dem_nums.extend([0] * (req_size - len(dem_nums)))
        DEM_rap_arr = np.array(dem_nums[:req_size]).reshape((R, A+1, P))
        DEM_rap = {(r, a, p): DEM_rap_arr[r, a, p] for r in range(R) for a in range(A+1) for p in range(P)}
        M_rap = {(r, a): np.max(DEM_rap_arr[r, a, :]) for r in range(R) for a in range(A+1)}

        # --- 6. Upgrades ---
        upg_data = get_clean_matrix('Upgrades')
        UPG_g1g2 = {(g1, g2): int(upg_data[g1, g2]) if upg_data.size >= G*G else (1 if g1 == g2 else 0)
                    for g1 in range(G) for g2 in range(G)}

        # --- 7. Transfer Costs ---
        tc_data = get_clean_matrix('TransferCosts')
        TC_gs1s2 = {}
        if G == 1 and tc_data.shape >= (S, S):
            for s1 in range(S):
                for s2 in range(S): TC_gs1s2[(0, s1, s2)] = float(tc_data[s1, s2]) 
        else:
            for g in range(G):
                for s1 in range(S):
                    for s2 in range(S): TC_gs1s2[(g, s1, s2)] = 0.0

        # --- 8. Transfer Times ---
        tt_data = get_clean_matrix('TransferTimes')
        TT_s1s2 = {(s1, s2): int(tt_data[s1, s2]) for s1 in range(S) for s2 in range(S)}

        # --- 9. Lookups & Initial Conditions ---
        rental_gr = {r: r_data[r, 4] - 1 for r in range(R)}
        INX_gs = {(g, s): 0.0 for g in range(G) for s in range(S)}
        
        # Add ONY and ONU, defaulting to 0 as in paper
        ONY_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONY_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})
        ONU_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONU_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})

        print("Data loaded successfully.")

    except Exception as e:
        print(f"\n>>> FAILED to load data | Error: {str(e)}")
        return

    # ==========================================================================
    # --- STEP 2: BUILD AND SOLVE THE LINEAR (MILP) MODEL ---
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: BUILDING AND SOLVING LINEAR (MILP) MODEL")
    print(f"{'='*60}")
    
    model_lin = ConcreteModel()
    
    # ----- SETS -----
    model_lin.G = RangeSet(0, G-1)
    model_lin.S = RangeSet(0, S-1)
    model_lin.R = RangeSet(0, R-1)
    model_lin.A = RangeSet(0, A)
    model_lin.P = RangeSet(0, P-1)
    model_lin.T = RangeSet(0, T)
    model_lin.T_minus = RangeSet(0, T-1)

    # ----- Decision Variables (from linear.py) -----
    model_lin.w_O = Var(model_lin.G, model_lin.S, domain=NonNegativeIntegers)
    model_lin.w_L = Var(model_lin.G, model_lin.T_minus, model_lin.S,domain=NonNegativeIntegers)
    model_lin.q = Var(model_lin.R, model_lin.A, model_lin.P, domain=Binary)
    model_lin.x_L = Var(model_lin.G, model_lin.T, model_lin.S, domain=NonNegativeIntegers)
    model_lin.x_O = Var(model_lin.G, model_lin.T, model_lin.S, domain=NonNegativeIntegers)
    model_lin.y_L = Var(model_lin.S, model_lin.S, model_lin.G, model_lin.T_minus, domain=NonNegativeIntegers)
    model_lin.y_O = Var(model_lin.S, model_lin.S, model_lin.G, model_lin.T_minus, domain=NonNegativeIntegers)
    model_lin.u_L = Var(model_lin.R, model_lin.A, model_lin.G, domain=NonNegativeIntegers)
    model_lin.u_O = Var(model_lin.R, model_lin.A, model_lin.G, domain=NonNegativeIntegers)
    model_lin.f_L = Var(model_lin.G, model_lin.T, domain=NonNegativeIntegers)
    model_lin.f_O = Var(model_lin.G, model_lin.T, domain=NonNegativeIntegers)
    # --- Linearization Variables ---
    model_lin.U = Var(model_lin.R, model_lin.A, domain=NonNegativeIntegers)
    model_lin.v = Var(model_lin.R, model_lin.A, model_lin.P, domain=NonNegativeIntegers)

    # ----- Objective Function (LINEAR) -----
    model_lin.obj = Objective(sense=maximize, rule=lambda m:
                          sum(m.v[r, a, p] * PRI_pg[p, rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
                          (sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) +
                           sum(m.f_L[g, t]*LEA_g[g] for g in m.G for t in m.T_minus) +
                           sum(m.f_O[g, t]*OWN_g[g] for g in m.G for t in m.T_minus) +
                           sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
                           sum((m.u_L[r, a, g]+m.u_O[r, a, g])*PYU for g in m.G for r in m.R for a in m.A if rental_gr[r] != g)))

    # ----- Linearization Constraints -----
    model_lin.c1 = Constraint(model_lin.R, model_lin.A, rule=lambda m, r, a: m.U[r, a] == sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G))
    model_lin.c2 = Constraint(model_lin.R, model_lin.A, model_lin.P, rule=lambda m, r, a, p: m.v[r, a, p] <= M_rap[r, a]*m.q[r, a, p])
    model_lin.c3 = Constraint(model_lin.R, model_lin.A, model_lin.P, rule=lambda m, r, a, p: m.v[r, a, p] <= m.U[r, a])
    model_lin.c4 = Constraint(model_lin.R, model_lin.A, model_lin.P, rule=lambda m, r, a, p: m.v[r, a, p] >= m.U[r, a] - M_rap[r, a]*(1-m.q[r, a, p]))

    # ----- Other Constraints (Corrected per Paper) -----
    def stock_O(m, g, t, s):
        if t == 0:
            return m.x_O[g, 0, s] == INX_gs[g, s] + m.w_O[g, s]
        rin = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
        rout = sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
        tin = sum(m.y_O[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
        tout = sum(m.y_O[s, s2, g, t] for s2 in m.S if t in m.T_minus)
        ony = ONY_gts[g, t, s, 'O']
        onu = ONU_gts[g, t, s, 'O']
        return m.x_O[g, t, s] == m.x_O[g, t-1, s] + ony + onu + rin - rout + tin - tout
    model_lin.c_stockO = Constraint(model_lin.G, model_lin.T, model_lin.S, rule=stock_O)

    def stock_L(m, g, t, s):
        if t == 0:
            return m.x_L[g, 0, s] == 0
        rin = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 1] == s+1 and r_data[r, 3] == t)
        rout = sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t)
        tin = sum(m.y_L[s2, s, g, t-TT_s1s2[s2, s]] for s2 in m.S if t-TT_s1s2[s2, s] in m.T_minus)
        tout = sum(m.y_L[s, s2, g, t] for s2 in m.S if t in m.T_minus)
        acq = m.w_L[g, t-1, s] if (t-1) in m.T_minus else 0
        ony = ONY_gts[g, t, s, 'L']
        onu = ONU_gts[g, t, s, 'L']
        if t <= LP_g[g]:
            return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ony + onu + acq 
                                     + rin - rout + tin - tout)
        else:
            ret_index = t - LP_g[g] - 1
            ret = m.w_L[g, ret_index, s] if ret_index in m.T_minus else 0
            return m.x_L[g, t, s] == (m.x_L[g, t-1, s] + ony + onu + acq - ret 
                                     + rin - rout + tin - tout)
    model_lin.c_stockL = Constraint(model_lin.G, model_lin.T, model_lin.S, rule=stock_L)

    model_lin.c_cap = Constraint(model_lin.G, model_lin.T_minus, model_lin.S, rule=lambda m, g, t, s:
                             sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
                             sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])
    model_lin.c_dem = Constraint(model_lin.R, model_lin.A, model_lin.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
    model_lin.c_price = Constraint(model_lin.R, model_lin.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
    model_lin.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)
    model_lin.c_upg = Constraint(model_lin.R, model_lin.A, model_lin.G, rule=lambda m, r, a, g: m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)
    model_lin.c_fL = Constraint(model_lin.G, model_lin.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
    model_lin.c_fO = Constraint(model_lin.G, model_lin.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))

    # ----- SOLVE LINEAR MODEL -----
    print("\n[Gurobi output for LINEAR model...]")
    opt = SolverFactory('gurobi')
    opt.options['TimeLimit'] = 600
    res_lin = opt.solve(model_lin, tee=True)
    
    linear_obj_value = 0.0
    solution = {}
    
    if res_lin.solver.termination_condition == TerminationCondition.optimal:
        print("\nLinear model solved to optimality.")
        linear_obj_value = value(model_lin.obj)
        print(f"LINEARIZED Objective Value: {linear_obj_value:.4f}")
        
        print("\nExtracting FULL solution from linear model...")
        solution['w_O'] = model_lin.w_O.get_values()
        solution['w_L'] = model_lin.w_L.get_values()
        solution['q'] = model_lin.q.get_values()
        solution['x_L'] = model_lin.x_L.get_values()
        solution['x_O'] = model_lin.x_O.get_values()
        solution['y_L'] = model_lin.y_L.get_values()
        solution['y_O'] = model_lin.y_O.get_values()
        solution['u_L'] = model_lin.u_L.get_values()
        solution['u_O'] = model_lin.u_O.get_values()
        solution['f_L'] = model_lin.f_L.get_values()
        solution['f_O'] = model_lin.f_O.get_values()
        print(f"Extracted all variable values.")
    
    else:
        print(f"Linear model solve FAILED. Status: {res_lin.solver.termination_condition}")
        return

    # ==========================================================================
    # --- STEP 3: BUILD AND *CHECK* THE NON-LINEAR (MINLP) MODEL ---
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: BUILDING NON-LINEAR (MINLP) MODEL WITH FIXED SOLUTION")
    print(f"{'='*60}")
    
    model_nl = ConcreteModel()

    # ----- SETS -----
    model_nl.G = RangeSet(0, G-1)
    model_nl.S = RangeSet(0, S-1)
    model_nl.R = RangeSet(0, R-1)
    model_nl.A = RangeSet(0, A)
    model_nl.P = RangeSet(0, P-1)
    model_nl.T = RangeSet(0, T)
    model_nl.T_minus = RangeSet(0, T-1)

    # ----- Decision Variables (from nonlinear.py) -----
    model_nl.w_O = Var(model_nl.G, model_nl.S, domain=NonNegativeIntegers)
    model_nl.w_L = Var(model_nl.G, model_nl.T_minus, model_nl.S, domain=NonNegativeIntegers)
    model_nl.q = Var(model_nl.R, model_nl.A, model_nl.P, domain=Binary)
    model_nl.x_L = Var(model_nl.G, model_nl.T, model_nl.S, domain=NonNegativeIntegers)
    model_nl.x_O = Var(model_nl.G, model_nl.T, model_nl.S, domain=NonNegativeIntegers)
    model_nl.y_L = Var(model_nl.S, model_nl.S, model_nl.G, model_nl.T_minus, domain=NonNegativeIntegers)
    model_nl.y_O = Var(model_nl.S, model_nl.S, model_nl.G, model_nl.T_minus, domain=NonNegativeIntegers)
    model_nl.u_L = Var(model_nl.R, model_nl.A, model_nl.G, domain=NonNegativeIntegers)
    model_nl.u_O = Var(model_nl.R, model_nl.A, model_nl.G, domain=NonNegativeIntegers)
    model_nl.f_L = Var(model_nl.G, model_nl.T, domain=NonNegativeIntegers)
    model_nl.f_O = Var(model_nl.G, model_nl.T, domain=NonNegativeIntegers)

    # ----- Objective Function (NON-LINEAR) -----
    print("Setting NON-LINEAR objective function...")
    def obj_rule_nl(m):
        revenue = sum((m.u_L[r, a, g] + m.u_O[r, a, g]) * m.q[r, a, p] * PRI_pg[p, rental_gr[r]]
                      for r in m.R for a in m.A for g in m.G for p in m.P)
        buy_cost = sum(m.w_O[g, s] * COS_g[g] for g in m.G for s in m.S)
        lease_cost = sum(m.f_L[g, t] * LEA_g[g] for g in m.G for t in m.T_minus)
        own_cost = sum(m.f_O[g, t] * OWN_g[g] for g in m.G for t in m.T_minus)
        transfer_cost = sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2]
                           for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus)
        upgrade_cost = sum((m.u_L[r, a, g] + m.u_O[r, a, g]) * PYU
                          for g in m.G for r in m.R for a in m.A
                          if rental_gr[r] != g)
        return revenue - buy_cost - lease_cost - own_cost - transfer_cost - upgrade_cost
    model_nl.obj = Objective(rule=obj_rule_nl, sense=maximize)
    
    # ----- Constraints (Identical to linear model) -----
    model_nl.c_stockO = Constraint(model_nl.G, model_nl.T, model_nl.S, rule=stock_O)
    model_nl.c_stockL = Constraint(model_nl.G, model_nl.T, model_nl.S, rule=stock_L)
    model_nl.c_cap = Constraint(model_nl.G, model_nl.T_minus, model_nl.S, rule=lambda m, g, t, s:
                             sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
                             sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])
    model_nl.c_dem = Constraint(model_nl.R, model_nl.A, model_nl.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
    model_nl.c_price = Constraint(model_nl.R, model_nl.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
    model_nl.c_upg = Constraint(model_nl.R, model_nl.A, model_nl.G, rule=lambda m, r, a, g: m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)
    model_nl.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)
    model_nl.c_fL = Constraint(model_nl.G, model_nl.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
    model_nl.c_fO = Constraint(model_nl.G, model_nl.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))


    # ----- *** FIX ALL VARIABLES *** -----
    print("\nFixing ALL variables in non-linear model to linear solution...")
    
    def fix_vars(model_var, solution_dict):
        for idx, val in solution_dict.items():
            model_var[idx].fix(val)
            
    fix_vars(model_nl.w_O, solution['w_O'])
    fix_vars(model_nl.w_L, solution['w_L'])
    fix_vars(model_nl.q, solution['q'])
    fix_vars(model_nl.x_L, solution['x_L'])
    fix_vars(model_nl.x_O, solution['x_O'])
    fix_vars(model_nl.y_L, solution['y_L'])
    fix_vars(model_nl.y_O, solution['y_O'])
    fix_vars(model_nl.u_L, solution['u_L'])
    fix_vars(model_nl.u_O, solution['u_O'])
    fix_vars(model_nl.f_L, solution['f_L'])
    fix_vars(model_nl.f_O, solution['f_O'])

    print("All variables fixed.")
    
    # ----- "SOLVE" (CHECK FEASIBILITY) OF NON-LINEAR MODEL -----
    print("\n[Gurobi output for NON-LINEAR model (checking feasibility)...]")
    solver_nl = SolverFactory('gurobi')
    solver_nl.options['NonConvex'] = 2 # Not needed, but good practice
    
    res_nl = solver_nl.solve(model_nl, tee=True)

    nonlinear_obj_value = 0.0
    
    print("\nFeasibility Check Results:")
    print(f"  Solver Status: {res_nl.solver.status}")
    print(f"  Termination Condition: {res_nl.solver.termination_condition}")

    if res_nl.solver.termination_condition == TerminationCondition.optimal or \
       res_nl.solver.termination_condition == TerminationCondition.feasible:
        
        print("\n*** FEASIBILITY CONFIRMED ***")
        print("The linear solution is 100% feasible in the non-linear model.")
        
        nonlinear_obj_value = value(model_nl.obj)
        print(f"NON-LINEAR Objective Value: {nonlinear_obj_value:.4f}")
    else:
        print("\n*** FEASIBILITY FAILED ***")
        print("The linear solution is NOT feasible in the non-linear model.")
        print("This indicates a mismatch in the constraint definitions.")
        return

    # ==========================================================================
    # --- STEP 4: FINAL COMPARISON ---
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 4: FINAL VALIDATION RESULTS")
    print(f"{'='*80}")
    
    print(f"Linearized (MILP) Objective Value:    {linear_obj_value:>20.4f}")
    print(f"Non-Linear (Fixed) Objective Value: {nonlinear_obj_value:>20.4f}")
    
    difference = abs(linear_obj_value - nonlinear_obj_value)
    print(f"Absolute Difference:                  {difference:>20.4f}")
    
    if difference < 1e-3: # Use a small tolerance for floating point math
        print("\n*** VALIDATION SUCCESSFUL ***")
        print("The objective values are identical. The linearization is correct.")
    else:
        print("\n*** VALIDATION FAILED ***")
        print("The objective values differ, check the linearization constraints.")
        

# ==============================================================================
# BATCH LOOP
# ==============================================================================
if __name__ == "__main__":
    
    # Run ONLY for Instance 1
    fname = r"data\Inst1.xlsx" # Use 'r' for raw string to handle backslash
    
    if not os.path.exists(fname):
        print(f"ERROR: Cannot find file at {fname}")
        print("Please make sure the 'data' folder is in the same directory as this script,")
        print("and it contains 'Inst1.xlsx'.")
    else:
        validate_full_solution(fname)

    print("\nBatch run finished.")