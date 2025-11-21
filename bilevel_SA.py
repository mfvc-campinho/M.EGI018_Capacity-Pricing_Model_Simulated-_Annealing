# ==============================================================================
"""
[M.EGI018] Operations Management Project

SIMULATED ANNEALING (SA) - DETAILED LOGGING VERSION
---------------------------------------------------
- Explicitly logs: "ACCEPTED (BETTER)", "ACCEPTED (WORSE)", "REJECTED".
- Shows detailed timing: Neighbor Gen (s) vs Gurobi Eval (s).
- Uses Aggressive Capacity Heuristic.
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
# 1. DATA LOADING
# ==============================================================================
class InstanceData:
    def __init__(self, filename):
        print(f"Loading Instance Data from {filename}...")
        xls = pd.ExcelFile(filename)
        
        def get_mat(sheet):
            df = pd.read_excel(xls, sheet, header=None)
            return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(how='all', axis=1).values

        up = pd.read_excel(xls, 'UnitParameters', header=None, index_col=0)
        self.G = int(up.loc['G'].iloc[0]); self.S = int(up.loc['S'].iloc[0])
        self.A = int(up.loc['A'].iloc[0]); self.T = int(up.loc['T'].iloc[0])
        self.PYU = float(up.loc['PYU'].iloc[0]); self.BUD = float(up.loc['BUD'].iloc[0])

        rd = pd.read_excel(xls, 'RentalTypes')
        r_pot = rd.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(r_pot)==0: r_pot = rd.iloc[:, 0:5].apply(pd.to_numeric, errors='coerce').dropna(how='any')
        self.r_data = r_pot.values.astype(int)
        self.R = len(self.r_data)
        self.rental_gr = {r: self.r_data[r, 4] - 1 for r in range(self.R)}

        pbg = pd.read_excel(xls, 'ParametersByGroup', header=None, index_col=0)
        def gp(n, t=float): 
            v = pbg.loc[n].values.flatten(); 
            return {g: t(v[g]) if g < len(v) else t(v[0]) for g in range(self.G)}
        self.LEA_g = gp('LEA_g'); self.LP_g = gp('LP_g', int)
        self.COS_g = gp('COS_g'); self.OWN_g = gp('OWN_g')

        pd_mat = get_mat('Prices')
        self.P = pd_mat.shape[0]
        self.PRI_pg = {(p, g): pd_mat[p, g] for p in range(self.P) for g in range(self.G)}

        dr = pd.read_excel(xls, 'Demand', header=None).values.flatten()
        dn = [x for x in dr if isinstance(x, (int, float)) and not np.isnan(x)]
        self.DEM_rap = {}
        self.M_rap = {}
        self.demand_weights_gs = {} 
        idx = 0
        max_idx = len(dn)
        for r in range(self.R):
            s_start = self.r_data[r, 0] - 1
            g_req = self.rental_gr[r]
            for a in range(self.A + 1):
                max_val = 0
                for p in range(self.P):
                    val = dn[idx] if idx < max_idx else 0
                    self.DEM_rap[(r, a, p)] = val
                    if val > max_val: max_val = val
                    idx += 1
                self.M_rap[(r, a)] = max_val
                key = (g_req, s_start)
                self.demand_weights_gs[key] = self.demand_weights_gs.get(key, 0) + max_val
        self.gs_keys = list(self.demand_weights_gs.keys())
        self.gs_vals = list(self.demand_weights_gs.values())

        ud = get_mat('Upgrades')
        self.UPG_g1g2 = {}
        for g1 in range(self.G):
            for g2 in range(self.G):
                try: self.UPG_g1g2[(g1, g2)] = int(ud[g1, g2])
                except: self.UPG_g1g2[(g1, g2)] = 1 if g1==g2 else 0

        tc = get_mat('TransferCosts'); tt = get_mat('TransferTimes')
        self.TC_gs1s2 = {}; self.TT_s1s2 = {}
        for s1 in range(self.S):
            for s2 in range(self.S):
                self.TT_s1s2[(s1, s2)] = int(tt[s1, s2])
                for g in range(self.G):
                    self.TC_gs1s2[(g, s1, s2)] = float(tc[s1, s2]) if self.G==1 or tc.shape==(self.S,self.S) else 0.0

        self.INX_gs = {(g, s): 0.0 for g in range(self.G) for s in range(self.S)}

# ==============================================================================
# 2. PERSISTENT EVALUATOR
# ==============================================================================
class PersistentEvaluator:
    def __init__(self, data):
        print("Building Persistent Model...")
        self.data = data
        self.model = ConcreteModel()
        m = self.model
        m.G = RangeSet(0, data.G-1); m.S = RangeSet(0, data.S-1)
        m.R = RangeSet(0, data.R-1); m.A = RangeSet(0, data.A)
        m.P = RangeSet(0, data.P-1); m.T = RangeSet(0, data.T)
        m.T_minus = RangeSet(0, data.T-1)

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

        def obj_rule(mod):
            rev = sum(mod.v[r, a, p] * data.PRI_pg[p, data.rental_gr[r]] for r in mod.R for a in mod.A for p in mod.P)
            cost_buy = sum(mod.w_O[g, s]*data.COS_g[g] for g in mod.G for s in mod.S)
            cost_lease = sum(mod.f_L[g, t]*data.LEA_g[g] for g in mod.G for t in mod.T_minus)
            cost_own = sum(mod.f_O[g, t]*data.OWN_g[g] for g in mod.G for t in mod.T_minus)
            cost_trans = sum((mod.y_L[s1, s2, g, t] + mod.y_O[s1, s2, g, t]) * data.TC_gs1s2[g, s1, s2] for s1 in mod.S for s2 in mod.S for g in mod.G for t in mod.T_minus)
            cost_upg = sum((mod.u_L[r, a, g]+mod.u_O[r, a, g])*data.PYU for g in mod.G for r in mod.R for a in mod.A if data.rental_gr[r] != g)
            return rev - cost_buy - cost_lease - cost_own - cost_trans - cost_upg
        m.obj = Objective(rule=obj_rule, sense=maximize)

        m.c_U = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.v[r, a, p] for p in mod.P))
        m.c1 = Constraint(m.R, m.A, rule=lambda mod, r, a: mod.U[r, a] == sum(mod.u_L[r, a, g]+mod.u_O[r, a, g] for g in mod.G))
        
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

        print("Initializing Gurobi Persistent...")
        self.opt = SolverFactory('gurobi_persistent')
        self.opt.set_instance(self.model)
        self.opt.options['OutputFlag'] = 0 

    def update_and_solve(self, w_owned, w_leased, pricing_policy):
        m = self.model
        data = self.data

        # Fix Capacity
        for g in m.G:
            for s in m.S:
                val = w_owned.get((g,s), 0)
                if m.w_O[g,s].value != val:
                    m.w_O[g,s].fix(val)
                    self.opt.update_var(m.w_O[g,s])

        for g in m.G:
            for t in m.T_minus:
                for s in m.S:
                    val = w_leased.get((g,t,s), 0)
                    if m.w_L[g,t,s].value != val:
                        m.w_L[g,t,s].fix(val)
                        self.opt.update_var(m.w_L[g,t,s])
        
        # Update Pricing
        for r in m.R:
            for a in m.A:
                target = pricing_policy.get((r, a), -1)
                for p in m.P:
                    ub = data.DEM_rap[r, a, p] if p == target else 0
                    if m.v[r,a,p].ub != ub:
                        m.v[r,a,p].setub(ub)
                        self.opt.update_var(m.v[r,a,p])

        res = self.opt.solve(m, save_results=False, load_solutions=False)
        if res.solver.termination_condition == TerminationCondition.optimal:
            self.opt.load_vars()
            return value(m.obj)
        else:
            return -1e9

# ==============================================================================
# 3. HEURISTICS
# ==============================================================================
def load_initial_capacity_from_excel(excel_path, sheet_owned, sheet_leased):
    # return {}, {}
    if not os.path.exists(excel_path): return {}, {}
    try:
        xls = pd.ExcelFile(excel_path)
        df_o = pd.read_excel(xls, sheet_owned)
        df_l = pd.read_excel(xls, sheet_leased)
        w_owned = {}
        for _, row in df_o.iterrows():
            val = float(row['w_owned'])
            if val > 0: w_owned[(int(row['group'])-1, int(row['station'])-1)] = val
        w_leased = {}
        for _, row in df_l.iterrows():
            val = float(row['w_leased'])
            if val > 0: w_leased[(int(row['group'])-1, int(row['time']), int(row['station'])-1)] = val
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

def perturb_capacity_smart(w_owned_in, w_leased_in, data):
    w_owned = copy.deepcopy(w_owned_in)
    w_leased = copy.deepcopy(w_leased_in)
    roll = random.random()
    
    if roll < 0.4: # ADD (50-200 units)
        if data.gs_keys:
            (g, s) = random.choices(data.gs_keys, weights=data.gs_vals, k=1)[0]
            add_amount = random.randint(50, 200)
            if random.random() < 0.5:
                w_owned[(g, s)] = w_owned.get((g, s), 0) + add_amount
            else:
                t = random.randint(0, data.T-1)
                w_leased[(g, t, s)] = w_leased.get((g, t, s), 0) + add_amount
                
    elif roll < 0.8: # REMOVE (5-15%)
        if w_owned and random.random() < 0.5:
            k = random.choice(list(w_owned.keys()))
            remove_amount = max(1, int(w_owned[k] * random.uniform(0.05, 0.15)))
            w_owned[k] = max(0, w_owned[k] - remove_amount)
            if w_owned[k] == 0: del w_owned[k]
        elif w_leased:
            k = random.choice(list(w_leased.keys()))
            remove_amount = max(1, int(w_leased[k] * random.uniform(0.05, 0.15)))
            w_leased[k] = max(0, w_leased[k] - remove_amount)
            if w_leased[k] == 0: del w_leased[k]
            
    else: # SHAKE (+/- 5%)
        if w_owned:
            k = random.choice(list(w_owned.keys()))
            shake_amount = max(1, int(w_owned[k] * 0.05))
            w_owned[k] = max(0, w_owned[k] + random.randint(-shake_amount, shake_amount))
        if w_leased:
            k = random.choice(list(w_leased.keys()))
            shake_amount = max(1, int(w_leased[k] * 0.05))
            w_leased[k] = max(0, w_leased[k] + random.randint(-shake_amount, shake_amount))
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

def save_solution_to_excel(w_owned, w_leased, pricing, filename):
    print(f"Saving best solution to {filename}...")
    o_data = []
    for (g, s), val in w_owned.items():
        if val > 0: o_data.append({'Group': g+1, 'Station': s+1, 'Quantity': val})
    l_data = []
    for (g, t, s), val in w_leased.items():
        if val > 0: l_data.append({'Group': g+1, 'Time': t, 'Station': s+1, 'Quantity': val})
    p_data = []
    for (r, a), p in pricing.items(): p_data.append({'RentalID': r, 'Antecedence': a, 'PriceLevel': p})
    
    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(o_data).to_excel(writer, sheet_name='Owned_Capacity', index=False)
        pd.DataFrame(l_data).to_excel(writer, sheet_name='Leased_Capacity', index=False)
        pd.DataFrame(p_data).to_excel(writer, sheet_name='Pricing_Policy', index=False)
    print("Save complete.")

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def run_sa_detailed(instance, heuristic_file, sheet_o, sheet_l, max_seconds=120):
    print(f"Running SA for {instance} (Limit: {max_seconds}s)")
    
    data = InstanceData(instance)
    evaluator = PersistentEvaluator(data)
    
    curr_w_o, curr_w_l = load_initial_capacity_from_excel(heuristic_file, sheet_o, sheet_l)
    curr_price = generate_greedy_pricing(data)
    
    print("Evaluating Initial...")
    curr_obj = evaluator.update_and_solve(curr_w_o, curr_w_l, curr_price)
    print(f"Initial Profit: {curr_obj:,.0f}")
    
    best_obj = curr_obj
    best_w_o = copy.deepcopy(curr_w_o)
    best_w_l = copy.deepcopy(curr_w_l)
    best_price = copy.deepcopy(curr_price)
    
    T = 5000; alpha = 0.95; PHASE_LENGTH = 10
    start_time = time.time()
    total_iter = 0
    
    # Columns for detailed logging
    print("\nIter  | Phase     | Temp       | New Obj         | Best Obj        | Neigh(s) | Eval(s)  | Status")
    print("-" * 110)
    
    while (time.time() - start_time) < max_seconds:
        total_iter += 1
        
        phase_idx = (total_iter - 1) // PHASE_LENGTH
        is_capacity_phase = (phase_idx % 2 == 0)
        phase_name = "CAPACITY" if is_capacity_phase else "PRICING"
        
        t0 = time.time()
        if is_capacity_phase:
            nw_o, nw_l = perturb_capacity_smart(curr_w_o, curr_w_l, data)
            n_price = curr_price 
        else:
            nw_o, nw_l = curr_w_o, curr_w_l
            n_price = perturb_pricing(curr_price, data.P, mutation_rate=0.01)
        t1 = time.time()
        
        n_obj = evaluator.update_and_solve(nw_o, nw_l, n_price)
        t2 = time.time()
        
        delta = n_obj - curr_obj
        accepted = False
        if delta > 0:
            accepted = True
        elif n_obj > -1e8: 
            try: prob = math.exp(delta / T)
            except: prob = 0
            if random.random() < prob: accepted = True
        
        status_msg = ""
        if accepted:
            curr_w_o, curr_w_l, curr_price = nw_o, nw_l, n_price
            curr_obj = n_obj
            status_msg = "ACCEPTED (WORSE)" if delta < 0 else "ACCEPTED (BETTER)"
            
            if curr_obj > best_obj:
                best_obj = curr_obj
                best_w_o = copy.deepcopy(curr_w_o)
                best_w_l = copy.deepcopy(curr_w_l)
                best_price = copy.deepcopy(curr_price)
                status_msg = "NEW BEST"
        
        neigh_time = t1 - t0
        eval_time = t2 - t1
        elapsed = time.time() - start_time
        
        # Only print every iteration or interesting ones to keep log readable
        print(f"{total_iter:<5} | {phase_name:<9} | {T:<10.2f} | {n_obj:<15,.0f} | {best_obj:<15,.0f} | {neigh_time:<8.4f} | {eval_time:<8.4f} | {status_msg}")
        
        if total_iter % 5 == 0:
            T *= alpha

    print(f"\nTime Limit Reached. Best Profit: {best_obj:,.0f}")
    save_solution_to_excel(best_w_o, best_w_l, best_price, "BestSolution_Inst40.xlsx")

if __name__ == "__main__":
    INSTANCE = r"data\Inst40.xlsx"
    HEURISTIC_EXCEL = r"heuristic_fleet_40_instances.xlsx"
    SHEET_OWNED = "Inst40_owned"
    SHEET_LEASED = "Inst40_leased"
    
    if os.path.exists(INSTANCE) and os.path.exists(HEURISTIC_EXCEL):
        run_sa_detailed(INSTANCE, HEURISTIC_EXCEL, SHEET_OWNED, SHEET_LEASED, max_seconds=120)
    else:
        print(f"Error: File not found.")