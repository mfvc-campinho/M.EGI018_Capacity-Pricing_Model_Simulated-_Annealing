# ==============================================================================
"""
[M.EGI018] Operations Management Project

SIMULATED ANNEALING (SA) - PERSISTENT SOLVER
--------------------------------------------
Performance Fix:
  - Uses 'gurobi_persistent' to keep model in memory.
  - Updates bounds/fixed values instead of rebuilding constraints.
  - EXPECTED SPEED: ~0.5s to 1.0s per iteration.
"""
# ==============================================================================
import os
import sys
import time
import math
import random
import copy
from pyomo.environ import *
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
import pandas as pd
import numpy as np

# ==============================================================================
# 1. DATA LOADING CLASS
# ==============================================================================
class InstanceData:
    def __init__(self, filename):
        print(f"Loading Instance Data from {filename}...")
        xls = pd.ExcelFile(filename)
        
        def get_mat(sheet):
            df = pd.read_excel(xls, sheet, header=None)
            return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(how='all', axis=1).values

        # Unit Params
        up = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        self.G = int(up.loc['G'].iloc[0]); self.S = int(up.loc['S'].iloc[0])
        self.A = int(up.loc['A'].iloc[0]); self.T = int(up.loc['T'].iloc[0])
        self.PYU = float(up.loc['PYU'].iloc[0]); self.BUD = float(up.loc['BUD'].iloc[0])

        # Rentals
        rd = pd.read_excel(xls, 'RentalTypes')
        r_pot = rd.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot)==0: r_pot = rd.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        self.r_data = r_pot.values.astype(int)
        self.R = len(self.r_data)
        self.rental_gr = {r: self.r_data[r, 4] - 1 for r in range(self.R)}

        # Groups
        pbg = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def gp(n, t=float): 
            v = pbg.loc[n].values.flatten(); 
            return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}
        self.LEA_g = gp('LEA_g'); self.LP_g = gp('LP_g', int)
        self.COS_g = gp('COS_g'); self.OWN_g = gp('OWN_g')

        # Prices
        pd_mat = get_mat('Prices')
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        # Demand
        dr = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dn = [x for x in dr if isinstance(x, (int, float)) and not np.isnan(x)]
        self.DEM_rap = {}
        self.M_rap = {}
        idx = 0
        max_idx = len(dn)
        for r in range(self.R):
            for a in range(self.A + 1):
                max_val = 0
                for p in range(self.P):
                    val = dn[idx] if idx < max_idx else 0
                    self.DEM_rap[(r, a, p)] = val
                    if val > max_val: max_val = val
                    idx += 1
                self.M_rap[(r, a)] = max_val

        # Upgrades
        ud = get_mat('Upgrades')
        self.UPG_g1g2 = {}
        for g1 in range(self.G):
            for g2 in range(self.G):
                try: self.UPG_g1g2[(g1, g2)] = int(ud[g1, g2])
                except: self.UPG_g1g2[(g1, g2)] = 1 if g1==g2 else 0

        # Transfer
        tc = get_mat('TransferCosts'); tt = get_mat('TransferTimes')
        self.TC_gs1s2 = {}; self.TT_s1s2 = {}
        for s1 in range(self.S):
            for s2 in range(self.S):
                self.TT_s1s2[(s1, s2)] = int(tt[s1, s2])
                for g in range(self.G):
                    self.TC_gs1s2[(g, s1, s2)] = float(tc[s1, s2]) if self.G==1 or tc.shape==(self.S,self.S) else 0.0

        self.INX_gs = {(g, s): 0.0 for g in range(self.G) for s in range(self.S)}
        print("Data Loaded.")

# ==============================================================================
# 2. PERSISTENT MODEL CLASS
# ==============================================================================

class PersistentEvaluator:
    def __init__(self, data):
        print("Building Persistent Model (This happens once)...")
        self.data = data
        self.model = ConcreteModel()
        
        # Sets
        m = self.model
        m.G = RangeSet(0, data.G-1); m.S = RangeSet(0, data.S-1)
        m.R = RangeSet(0, data.R-1); m.A = RangeSet(0, data.A)
        m.P = RangeSet(0, data.P-1); m.T = RangeSet(0, data.T)
        m.T_minus = RangeSet(0, data.T-1)

        # Vars
        m.w_O = Var(m.G, m.S, domain=NonNegativeReals)
        m.w_L = Var(m.G, m.T_minus, m.S, domain=NonNegativeReals)
        m.x_L = Var(m.G, m.T, m.S, domain=NonNegativeReals)
        m.x_O = Var(m.G, m.T, m.S, domain=NonNegativeReals)
        m.y_L = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
        m.y_O = Var(m.S, m.S, m.G, m.T_minus, domain=NonNegativeReals)
        m.u_L = Var(m.R, m.A, m.G, domain=NonNegativeReals)
        m.u_O = Var(m.R, m.A, m.G, domain=NonNegativeReals)
        m.f_L = Var(m.G, m.T, domain=NonNegativeReals)
        m.f_O = Var(m.G, m.T, domain=NonNegativeReals)
        m.U = Var(m.R, m.A, domain=NonNegativeReals)
        m.v = Var(m.R, m.A, m.P, domain=NonNegativeReals)

        # Objective
        def obj_rule(mod):
            rev = sum(mod.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] 
                      for r in mod.R for a in mod.A for p in mod.P)
            cost_buy = sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S)
            cost_lease = sum(mod.f_L[g, t]*data.LEA_g[g] for g in mod.G for t in mod.T_minus)
            cost_own = sum(mod.f_O[g, t]*data.OWN_g[g] for g in mod.G for t in mod.T_minus)
            cost_trans = sum((mod.y_L[s1, s2, g, t] + mod.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] 
                             for s1 in mod.S for s2 in mod.S for g in mod.G for t in mod.T_minus)
            cost_upg = sum((mod.u_L[r, a, g]+mod.u_O[r, a, g])*data.PYU 
                           for g in mod.G for r in mod.R for a in mod.A if data.rental_gr[r] != g)
            return rev - cost_buy - cost_lease - cost_own - cost_trans - cost_upg
        m.obj = Objective(rule=obj_rule, sense=maximize)

        # Constraints
        m.c_U = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.v[r, a, p] for p in mod.P))
        m.c1 = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for g in mod.G))
        
        # Stock Constraints
        def stock_O(mod, g, t, s):
            if t == 0: return mod.x_O[g, 0, s] == data.INX_gs[g, s] + mod.w_O[g, s]
            rin = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
            rout = sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
            tin = sum(mod.y_O[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
            tout = sum(mod.y_O[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
            return mod.x_O[g, t, s] == mod.x_O[g, t-1, s] + rin - rout + tin - tout
        m.c_stockO = Constraint(m.G, m.T, m.S, rule=stock_O)

        def stock_L(mod, g, t, s):
            if t == 0: return mod.x_L[g, 0, s] == 0
            rin = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 1] == s+1 and data.r_data[r, 3] == t)
            rout = sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t)
            tin = sum(mod.y_L[s2, s, g, t-data.TT_s1s2[s2, s]] for s2 in mod.S if t-data.TT_s1s2[s2, s] in mod.T_minus)
            tout = sum(mod.y_L[s, s2, g, t] for s2 in mod.S if t in mod.T_minus)
            acq = mod.w_L[g, t-1, s] if (t-1) in mod.T_minus else 0
            if t <= data.LP_g[g]: return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq + rin - rout + tin - tout
            ret_idx = t - data.LP_g[g] - 1
            ret = mod.w_L[g, ret_idx, s] if ret_idx in mod.T_minus else 0
            return mod.x_L[g, t, s] == mod.x_L[g, t-1, s] + acq - ret + rin - rout + tin - tout
        m.c_stockL = Constraint(m.G, m.T, m.S, rule=stock_L)

        m.c_cap = Constraint(m.G, m.T_minus, m.S, rule=lambda mod, g, t, s:
             sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 0] == s+1 and data.r_data[r, 2] == t) +
             sum(mod.y_L[s, s2, g, t]+mod.y_O[s, s2, g, t] for s2 in mod.S) <= mod.x_L[g, t, s] + mod.x_O[g, t, s])
        
        m.c_upg = Constraint(m.R, m.A, m.G, rule=lambda mod, r, a, g: 
                                 mod.u_L[r, a, g]+mod.u_O[r, a, g] == 0 if data.UPG_g1g2.get((data.rental_gr[r], g), 0) == 0 and data.rental_gr[r] != g else Constraint.Skip)
        m.c_fL = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_L[g, t] >= sum(mod.x_L[g, t, s] for s in mod.S) + sum(mod.u_L[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
        m.c_fO = Constraint(m.G, m.T, rule=lambda mod, g, t: mod.f_O[g, t] >= sum(mod.x_O[g, t, s] for s in mod.S) + sum(mod.u_O[r, a, g] for r in mod.R for a in mod.A if data.r_data[r, 2] <= t < data.r_data[r, 3]))
        m.c_bud = Constraint(rule=lambda mod: sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S) <= data.BUD)

        # INIT SOLVER
        print("Initializing Gurobi Persistent...")
        self.opt = SolverFactory('gurobi_persistent')
        self.opt.set_instance(self.model)
        self.opt.options['OutputFlag'] = 0 

    def update_and_solve(self, w_owned, w_leased, pricing_policy):
        m = self.model
        data = self.data

        # 1. Update Capacity (Fix Variables)
        for g in m.G:
            for s in m.S:
                val = w_owned.get((g,s), 0)
                if m.w_O[g,s].value != val:
                    m.w_O[g,s].fix(val)
                    # Note: gurobi_persistent handles variable bounds updates automatically, 
                    # but we specifically need to tell it if we change fixed status/value heavily.
                    self.opt.update_var(m.w_O[g,s])

        for g in m.G:
            for t in m.T_minus:
                for s in m.S:
                    val = w_leased.get((g,t,s), 0)
                    if m.w_L[g,t,s].value != val:
                        m.w_L[g,t,s].fix(val)
                        self.opt.update_var(m.w_L[g,t,s])
        
        # 2. Update Pricing (Change Bounds of 'v')
        # v[r,a,p] is 0 if p != target, else Demand
        for r in m.R:
            for a in m.A:
                target = pricing_policy.get((r, a), -1)
                for p in m.P:
                    if p == target:
                        # Allowed to sell
                        ub = data.DEM_rap[r, a, p]
                    else:
                        # Not allowed
                        ub = 0
                    
                    # Check if bound needs update to save API calls
                    if m.v[r,a,p].ub != ub:
                        m.v[r,a,p].setub(ub)
                        self.opt.update_var(m.v[r,a,p])

        # 3. Solve
        res = self.opt.solve(m, save_results=False, load_solutions=False)
        
        if res.solver.termination_condition == TerminationCondition.optimal:
            self.opt.load_vars() # Load values back to Pyomo model
            return value(m.obj)
        else:
            return -1e9

# ==============================================================================
# 3. SA HELPERS
# ==============================================================================

def load_initial_capacity(excel_path, sheet_owned, sheet_leased):
    if not os.path.exists(excel_path): return {}, {}
    try:
        xls_cap = pd.ExcelFile(excel_path)
        df_o = pd.read_excel(xls_cap, sheet_owned)
        df_l = pd.read_excel(xls_cap, sheet_leased)
        w_owned = {}
        for _, row in df_o.iterrows():
            if row['w_owned'] > 0:
                w_owned[(int(row['group'])-1, int(row['station'])-1)] = float(row['w_owned'])
        w_leased = {}
        for _, row in df_l.iterrows():
            if row['w_leased'] > 0:
                w_leased[(int(row['group'])-1, int(row['time']), int(row['station'])-1)] = float(row['w_leased'])
        return w_owned, w_leased
    except: return {}, {}

def generate_greedy_pricing(data):
    target_p = {}
    for r in range(data.R):
        for a in range(data.A + 1):
            best_rev = -1; best_p = 0
            for p in range(data.P):
                rev = data.PRI_pg[p, data.rental_gr[r]] * data.DEM_rap[r, a, p]
                if rev > best_rev: best_rev = rev; best_p = p
            target_p[(r, a)] = best_p
    return target_p

def perturb_capacity(w_owned_in, w_leased_in, range_val=1):
    w_owned = copy.deepcopy(w_owned_in); w_leased = copy.deepcopy(w_leased_in)
    if w_owned:
        k = random.choice(list(w_owned.keys()))
        w_owned[k] = max(0, w_owned[k] + random.randint(-range_val, range_val))
    if w_leased:
        k = random.choice(list(w_leased.keys()))
        w_leased[k] = max(0, w_leased[k] + random.randint(-range_val, range_val))
    return w_owned, w_leased

def perturb_pricing(pricing_in, P_max, mutation_rate=0.01):
    pricing = copy.deepcopy(pricing_in)
    num = max(1, int(len(pricing)*mutation_rate))
    targets = random.sample(list(pricing.keys()), num)
    for k in targets:
        opts = list(range(P_max)); 
        if pricing[k] in opts: opts.remove(pricing[k])
        if opts: pricing[k] = random.choice(opts)
    return pricing

# ==============================================================================
# 4. MAIN SA LOOP
# ==============================================================================

def run_sa(instance, heuristic):
    print(f"Running SA (Persistent Solver) for {instance}")
    
    # 1. Load Data & Init Solver
    data = InstanceData(instance)
    evaluator = PersistentEvaluator(data)
    
    # 2. Initial State
    curr_w_o, curr_w_l = load_initial_capacity(heuristic, "Inst40_owned", "Inst40_leased")
    curr_price = generate_greedy_pricing(data)
    
    print("Evaluating Initial...")
    curr_obj = evaluator.update_and_solve(curr_w_o, curr_w_l, curr_price)
    print(f"Initial Profit: {curr_obj:,.0f}")
    
    best_obj = curr_obj
    best_w_o = copy.deepcopy(curr_w_o)
    best_w_l = copy.deepcopy(curr_w_l)
    best_price = copy.deepcopy(curr_price)
    
    # Parameters
    T = 5000; T_min = 1; alpha = 0.90; iters_per_T = 5
    
    print("\nIter  | Temp       | New Obj         | Curr Obj        | Best Obj        | Neigh(s) | Eval(s)")
    print("-" * 95)
    
    total_iter = 0
    while T > T_min:
        for _ in range(iters_per_T):
            total_iter += 1
            print(f"Processing {total_iter}...", end='\r')
            
            t0 = time.time()
            nw_o, nw_l = perturb_capacity(curr_w_o, curr_w_l)
            n_price = perturb_pricing(curr_price, data.P, 0.01)
            t1 = time.time()
            
            # Evaluate (Fast Update)
            n_obj = evaluator.update_and_solve(nw_o, nw_l, n_price)
            t2 = time.time()
            
            # Accept?
            delta = n_obj - curr_obj
            accepted = False
            if delta > 0:
                accepted = True
            elif n_obj > -1e8: # Only consider if feasible
                try: prob = math.exp(delta / T)
                except: prob = 0
                if random.random() < prob: accepted = True
            
            status_tag = ""
            if accepted:
                curr_w_o, curr_w_l, curr_price = nw_o, nw_l, n_price
                curr_obj = n_obj
                if curr_obj > best_obj:
                    best_obj = curr_obj
                    status_tag = "NEW BEST"
            
            print(f"{total_iter:<5} | {T:<10.2f} | {n_obj:<15,.0f} | {curr_obj:<15,.0f} | {best_obj:<15,.0f} | {t1-t0:<8.4f} | {t2-t1:<8.4f} {status_tag}")
            
        T *= alpha

    print(f"\nDone. Best Profit: {best_obj:,.0f}")

if __name__ == "__main__":
    run_sa(r"data\Inst40.xlsx", r"heuristic_fleet_40_instances.xlsx")