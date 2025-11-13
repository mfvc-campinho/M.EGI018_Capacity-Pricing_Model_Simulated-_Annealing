# ==============================================================================
"""
[M.EGI018] Operations Management Project

CAR RENTAL FLEET MANAGEMENT - TIGHTENED LP (RELAXED) SOLVER
*** MODIFIED with a TIGHTER demand constraint ***

This script:
1. Imports a capacity solution (w_O, w_L) from heuristic.py
2. Fixes the w_O and w_L variables to those values.
3. Relaxes ALL variables (including q, x, y, u, f) to continuous.
4. USES A TIGHTER, WEIGHTED-AVERAGE DEMAND CONSTRAINT.
5. Solves the resulting LP.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
from pyomo.environ import *
import pandas as pd
import numpy as np
from constructive_heuristic import ConstructiveHeuristic # <-- IMPORTING THE HEURISTIC CLASS

# ==============================================================================
# CORE SOLVING FUNCTION
# ==============================================================================

# Function to solve a single instance of the car rental problem
def solve_instance(filename):
    print(f"\n{'='*60}")
    print(f"PROCESSING TIGHT LP RELAXATION (FIXED HEURISTIC): {filename}")
    print(f"{'='*60}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return {'Instance': filename, 'Status': 'File Not Found', 'Objective': None, 'Gap': None}

    # Read Excel file
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
        ONY_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONY_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})
        ONU_gts = {(g, t, s, 'L'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)}
        ONU_gts.update({(g, t, s, 'O'): 0.0 for g in range(G) for t in range(T+1) for s in range(S)})

        # =======================================
        # =================MODEL=================
        # =======================================
        
        print("\n*** RUNNING AS A PURE LP (ALL VARIABLES RELAXED) ***\n")

        model = ConcreteModel()
        model.G = RangeSet(0, G-1)
        model.S = RangeSet(0, S-1)
        model.R = RangeSet(0, R-1)
        model.A = RangeSet(0, A)
        model.P = RangeSet(0, P-1)
        model.T = RangeSet(0, T)
        model.T_minus = RangeSet(0, T-1)

        # ----- Decision Variables (RELAXED) -----
        model.w_O = Var(model.G, model.S, domain=NonNegativeReals)
        model.w_L = Var(model.G, model.T_minus, model.S, domain=NonNegativeReals)
        model.q = Var(model.R, model.A, model.P, domain=NonNegativeReals, bounds=(0, 1))
        model.x_L = Var(model.G, model.T, model.S, domain=NonNegativeReals)
        model.x_O = Var(model.G, model.T, model.S, domain=NonNegativeReals)
        model.y_L = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeReals)
        model.y_O = Var(model.S, model.S, model.G, model.T_minus, domain=NonNegativeReals)
        model.u_L = Var(model.R, model.A, model.G, domain=NonNegativeReals)
        model.u_O = Var(model.R, model.A, model.G, domain=NonNegativeReals)
        model.f_L = Var(model.G, model.T, domain=NonNegativeReals)
        model.f_O = Var(model.G, model.T, domain=NonNegativeReals)
        model.U = Var(model.R, model.A, domain=NonNegativeReals)
        model.v = Var(model.R, model.A, model.P, domain=NonNegativeReals)

        # ==================================================================
        # ========== FIXING CAPACITY FROM HEURISTIC ========================
        # ==================================================================
        print("\n*** MODIFICATION: Generating and fixing capacity from external heuristic... ***")
        heuristic = ConstructiveHeuristic()
        w_O_fixed, w_L_fixed = heuristic.generate_solution(
            G_set=model.G, 
            S_set=model.S, 
            T_minus_set=model.T_minus
        )
        print(f"Fixing {len(w_O_fixed)} w_O variables...")
        for (g, s), val in w_O_fixed.items():
            model.w_O[g, s].fix(val)
        print(f"Fixing {len(w_L_fixed)} w_L variables...")
        for (g, t, s), val in w_L_fixed.items():
            model.w_L[g, t, s].fix(val)
        print("*** Capacity variables w_O and w_L fixed from heuristic. ***\n")
        # ==================================================================

        # ----- Objective Function -----
        model.obj = Objective(sense=maximize, rule=lambda m:
                              sum(m.v[r, a, p] * PRI_pg[p, rental_gr[r]] for r in m.R for a in m.A for p in m.P) -
                              (sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) +
                               sum(m.f_L[g, t]*LEA_g[g] for g in m.G for t in m.T_minus) +
                               sum(m.f_O[g, t]*OWN_g[g] for g in m.G for t in m.T_minus) +
                               sum((m.y_L[s1, s2, g, t] + m.y_O[s1, s2, g, t]) * TC_gs1s2[g, s1, s2] for s1 in m.S for s2 in m.S for g in m.G for t in m.T_minus) +
                               sum((m.u_L[r, a, g]+m.u_O[r, a, g])*PYU for g in m.G for r in m.R for a in m.A if rental_gr[r] != g)))

        # ----- Constraints -----
        model.c1 = Constraint(model.R, model.A, rule=lambda m, r, a: m.U[r, a] == sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G))
        model.c2 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= M_rap[r, a]*m.q[r, a, p])
        model.c3 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] <= m.U[r, a])
        model.c4 = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: m.v[r, a, p] >= m.U[r, a] - M_rap[r, a]*(1-m.q[r, a, p]))

        # ==================================================================
        # ========== NEW TIGHTER DEMAND CONSTRAINT =========================
        # ==================================================================
        # We replace the original 'c_dem' with this tighter version.
        # This links total fulfillment 'U' to the weighted average of demand.
        
        # model.c_dem = Constraint(model.R, model.A, model.P, rule=lambda m, r, a, p: sum(m.u_L[r, a, g]+m.u_O[r, a, g] for g in m.G) <= DEM_rap[r, a, p] + (1-m.q[r, a, p])*M_rap[r, a])
        
        def tight_demand_rule(m, r, a):
            weighted_demand = sum(DEM_rap[r, a, p] * m.q[r, a, p] for p in m.P)
            return m.U[r, a] <= weighted_demand
            
        model.c_dem_tight = Constraint(model.R, model.A, rule=tight_demand_rule)
        
        print("--- Using TIGHTER weighted-average demand constraint ---")
        # ==================================================================

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
        model.c_stockO = Constraint(model.G, model.T, model.S, rule=stock_O)

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
        model.c_stockL = Constraint(model.G, model.T, model.S, rule=stock_L)

        model.c_cap = Constraint(model.G, model.T_minus, model.S, rule=lambda m, g, t, s:
                                 sum(m.u_L[r, a, g]+m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 0] == s+1 and r_data[r, 2] == t) +
                                 sum(m.y_L[s, s2, g, t]+m.y_O[s, s2, g, t] for s2 in m.S) <= m.x_L[g, t, s] + m.x_O[g, t, s])
        # Note: Original c_dem is replaced by c_dem_tight
        model.c_price = Constraint(model.R, model.A, rule=lambda m, r, a: sum(m.q[r, a, p] for p in m.P) == 1)
        model.c_bud = Constraint(rule=lambda m: sum(m.w_O[g, s]*COS_g[g] for g in m.G for s in m.S) <= BUD)
        model.c_upg = Constraint(model.R, model.A, model.G, rule=lambda m, r, a, g: m.u_L[r, a, g]+m.u_O[r, a, g] == 0 if UPG_g1g2.get((rental_gr[r], g), 0) == 0 and rental_gr[r] != g else Constraint.Skip)
        model.c_fL = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_L[g, t] >= sum(m.x_L[g, t, s] for s in m.S) + sum(m.u_L[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))
        model.c_fO = Constraint(model.G, model.T, rule=lambda m, g, t: m.f_O[g, t] >= sum(m.x_O[g, t, s] for s in m.S) + sum(m.u_O[r, a, g] for r in m.R for a in m.A if r_data[r, 2] <= t < r_data[r, 3]))

        # =================SOLVE=================
        print(f"\n[Gurobi Output for {filename} (LP, Fixed Fleet, Tight) follows...]")
        opt = SolverFactory('gurobi')
        opt.options['TimeLimit'] = 600
        
        res = opt.solve(model, tee=False) 

        # Extract results
        stat = str(res.solver.termination_condition)
        try:
            obj_val = value(model.obj)
            gap_str = "0.00%" # LPs don't have a MIP gap
        except Exception as e:
            obj_val = "N/A"
            gap_str = "Error"

        # Print summary
        print(f"\n>>> FINISHED {filename} | Obj: {obj_val} | Gap: {gap_str}")
        
        # --- (Optional) Print some solution details ---
        print("\n--- Sample LP Solution Results ---")
        print("Owned Fleet (w_O[g,s]) - (FIXED):")
        total_owned = 0
        for g in model.G:
            for s in model.S:
                val = value(model.w_O[g, s])
                total_owned += val
        print(f"  Total Owned (Fixed): {total_owned:.0f}")

        print("\nPricing (q[r,a,p]) - (Printing fractional values only):")
        count_frac = 0
        for r in model.R:
            for a in model.A:
                for p in model.P:
                    val = value(model.q[r,a,p])
                    if val > 0.01 and val < 0.99: # Print if fractional
                        if count_frac < 15: # Limit printout
                            print(f"  q[{r},{a},{p}] = {val:.2f}  <-- FRACTIONAL PRICE")
                        count_frac += 1
        if count_frac > 15:
            print(f"  ...and {count_frac - 15} more fractional prices.")
        elif count_frac == 0:
            print("  All q values are 0 or 1 (or close to it).")
        print("---------------------------------")
        
        return {'Instance': filename, 'Status': stat, 'Objective': obj_val, 'Gap': gap_str}

    except Exception as e:
        print(f"\n>>> FAILED {filename} | Error: {str(e)}")
        return {'Instance': filename, 'Status': 'Error', 'Objective': str(e), 'Gap': 'N/A'}


# ==============================================================================
# BATCH LOOP
# ==============================================================================
RESULTS_FILE = "LP_Tight_Fixed_Heuristic_Results.xlsx" # Changed output file name
results = []

print("Starting batch run (LP Tight, Fixed Heuristic, Instance 1 only)...")

# This loop runs ONLY for i=1
for i in range(1, 2):
    fname = r"data\Inst1.xlsx"

    # Solve
    res = solve_instance(fname)
    results.append(res)

    # Save immediately after each instance
    try:
        df_current = pd.DataFrame(results)
        df_current.to_excel(RESULTS_FILE, index=False)
        print(f">>> RESULTS UPDATED in {RESULTS_FILE}")
    except PermissionError:
        print(
            f"\nWARNING: Could not save to {RESULTS_FILE}. Is it open in Excel?")
        print("Continuing to next instance anyway...")

print("\nBatch run finished.")