import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import re

# --- 1. FUNÇÃO DE CARREGAMENTO DE DADOS (PARA LER VÁRIOS CSVs) ---

def load_data_from_inst_excel(excel_path="Inst2.xlsx"):
    """
    Lê todas as folhas relevantes de Inst2.xlsx e devolve o dicionário 'data'
    com o formato esperado pelo modelo Pyomo.
    """
    if not os.path.exists(excel_path):
        print(f"Erro: ficheiro '{excel_path}' não encontrado.")
        return None

    print(f"A carregar dados do ficheiro Excel: {os.path.basename(excel_path)}")
    data = {}
    xls = pd.ExcelFile(excel_path)

    # --- UnitParameters ---
    df_unit = pd.read_excel(xls, sheet_name='UnitParameters', header=None).dropna(how='all')
    unit_map = {str(row[0]).strip(): row[1] for _, row in df_unit.iterrows() if not pd.isna(row[1])}
    data['G'] = int(unit_map.get('G', 1))
    data['S'] = int(unit_map.get('S', 1))
    data['A'] = int(unit_map.get('A', 0))
    data['T'] = int(unit_map.get('T', 0))
    data['PYU'] = float(unit_map.get('PYU', 1.0))
    data['BUD'] = float(unit_map.get('BUD', 1e6))
    print("UnitParameters lido.")

    # --- RentalTypes ---
    df_r = pd.read_excel(xls, sheet_name='RentalTypes')
    df_r.columns = [str(c).strip().lower() for c in df_r.columns]
    if 'r' not in df_r.columns:
        df_r['r'] = range(1, len(df_r) + 1)
    df_r = df_r.set_index('r')
    for c in ['sout', 'sin', 'dout', 'din', 'gr']:
        if c not in df_r.columns:
            df_r[c] = 0
    data['R'] = len(df_r)
    data['sout'] = df_r['sout'].astype(int).to_dict()
    data['sin']  = df_r['sin'].astype(int).to_dict()
    data['dout'] = df_r['dout'].astype(int).to_dict()
    data['din']  = df_r['din'].astype(int).to_dict()
    data['gr']   = df_r['gr'].astype(int).to_dict()
    print(f"RentalTypes lido (R={data['R']}).")

    # --- ParametersByGroup ---
    df_pg = pd.read_excel(xls, sheet_name='ParametersByGroup', header=None).dropna(how='all')
    params = {str(row[0]).strip(): row[1] for _, row in df_pg.iterrows()}
    G = data['G']
    data['LEA'] = {g: float(params.get('LEA_g', 0.25)) for g in range(1, G+1)}
    data['LP']  = {g: int(params.get('LP_g', 8)) for g in range(1, G+1)}
    data['COS'] = {g: float(params.get('COS_g', 4.0)) for g in range(1, G+1)}
    data['OWN'] = {g: float(params.get('OWN_g', 0.02)) for g in range(1, G+1)}
    print("ParametersByGroup lido.")

    # --- Prices ---
    df_pr = pd.read_excel(xls, sheet_name='Prices', header=None)
    df_num = df_pr.apply(pd.to_numeric, errors='coerce')
    prices = df_num.select_dtypes(include='number').stack().dropna().tolist()
    P = len(prices)
    data['PRI'] = {(p_idx + 1, g): float(price) for p_idx, price in enumerate(prices) for g in range(1, data['G']+1)}
    data['P'] = P
    print(f"Prices lido. P={P}")

    # --- Demand ---
    df_dem_raw = pd.read_excel(xls, sheet_name='Demand', header=0)
    col_pa = []
    for col in df_dem_raw.columns:
        col_clean = str(col).strip().lower()
        m = re.search(r'p\s*=\s*(\d+)\s*,\s*a\s*=\s*(\d+)', col_clean)
        if m:
            p = int(m.group(1))
            a = int(m.group(2))
            col_pa.append((col, p, a))
    R_generated = range(1, df_dem_raw.shape[0] + 1)
    data['DEM'] = {}
    for idx_row, r in enumerate(R_generated):
        for (col, p, a) in col_pa:
            val = df_dem_raw.iloc[idx_row][col]
            if pd.isna(val): val = 0
            data['DEM'][(r, a, p)] = float(val)
    data['R'] = max(data['R'], len(R_generated))
    data['P'] = max(data['P'], max(p for (_, _, p) in col_pa))
    print(f"Demand lido (DEM entries): {len(data['DEM'])}")

    # --- Upgrades ---
    df_upg = pd.read_excel(xls, sheet_name='Upgrades', header=None)
    df_upg = df_upg.apply(pd.to_numeric, errors='coerce').fillna(0)

    # limitar à dimensão G x G
    G = data['G']
    df_upg = df_upg.iloc[:G, :G]

    # construir dicionário apenas com índices válidos
    data['UPG'] = {(i+1, j+1): float(df_upg.iloc[i, j]) for i in range(df_upg.shape[0]) for j in range(df_upg.shape[1])}
    print(f"Upgrades lido ({df_upg.shape[0]}x{df_upg.shape[1]}).")


    # --- TransferCosts ---
    df_tc = pd.read_excel(xls, sheet_name='TransferCosts', header=None)
    df_tc = df_tc.apply(pd.to_numeric, errors='coerce').fillna(0)

    S = data['S']
    G = data['G']

    # limitar à dimensão S x S
    df_tc = df_tc.iloc[:S, :S]

    # construir dicionário TC[g,s1,s2] só para índices válidos
    data['TC'] = {}
    for g in range(1, G+1):
        for i in range(df_tc.shape[0]):
            for j in range(df_tc.shape[1]):
                data['TC'][(g, i+1, j+1)] = float(df_tc.iloc[i, j])
    print(f"TransferCosts lido ({df_tc.shape[0]}x{df_tc.shape[1]}).")


    # --- TransferTimes ---
    df_tt = pd.read_excel(xls, sheet_name='TransferTimes', header=None)
    df_tt = df_tt.apply(pd.to_numeric, errors='coerce').fillna(0)

    S = data['S']
    df_tt = df_tt.iloc[:S, :S]  # limitar à dimensão S×S

    data['TT'] = {(i+1, j+1): float(df_tt.iloc[i, j]) for i in range(df_tt.shape[0]) for j in range(df_tt.shape[1])}
    print(f"TransferTimes lido ({df_tt.shape[0]}x{df_tt.shape[1]}).")

    # --- Parâmetros não presentes no Excel: inicialização padrão ---
    G, S, T = data['G'], data['S'], data['T']

    # Estoque inicial "owned"
    data['INX_O'] = {(g, s): 0.0 for g in range(1, G+1) for s in range(1, S+1)}

    # Novas unidades (Leased / Owned / New Units)
    data['ONY_L'] = {(g, t, s): 0.0 for g in range(1, G+1) for t in range(0, T+1) for s in range(1, S+1)}
    data['ONY_O'] = {(g, t, s): 0.0 for g in range(1, G+1) for t in range(0, T+1) for s in range(1, S+1)}
    data['ONU_L'] = {(g, t, s): 0.0 for g in range(1, G+1) for t in range(0, T+1) for s in range(1, S+1)}
    data['ONU_O'] = {(g, t, s): 0.0 for g in range(1, G+1) for t in range(0, T+1) for s in range(1, S+1)}

    print("Parâmetros adicionais (INX_O, ONY_*, ONU_*) inicializados por defeito.")

    print(f"✅ Carregamento concluído com sucesso: G={data['G']}, S={data['S']}, T={data['T']}, "
          f"A={data['A']}, P={data['P']}, R={data['R']}")
    return data

# --- 2. FUNÇÃO PARA CRIAR O MODELO MIP (LINEARIZADO) ---
# (Esta função está IDÊNTICA à anterior, não precisa de alterações)

def create_linearized_model(data, price_strategy):
    """
    Cria o modelo Pyomo LINEARIZADO.
    """
    print("A construir o modelo Pyomo LINEARIZADO (MIP)...")
    m = pyo.ConcreteModel(name="Car_Rental_Capacity_MIP")

    # --- 2.1. Conjuntos (Sets) [cite: 190-207] ---
    m.G = pyo.RangeSet(1, data['G'])
    m.S = pyo.RangeSet(1, data['S'])
    m.T = pyo.RangeSet(0, data['T'])
    m.R = pyo.RangeSet(1, data['R'])
    m.A = pyo.RangeSet(0, data['A'])
    m.P = pyo.RangeSet(1, data['P'])
    m.FleetType = pyo.Set(initialize=['L', 'O'])

    # Subconjuntos (Sets) [cite: 225-228]
    m.R_in = pyo.Set(m.S, m.T, within=m.R, initialize=lambda m, s, t: \
        [r for r in m.R if data['sin'][r] == s and data['din'][r] == t])
    m.R_out = pyo.Set(m.S, m.T, within=m.R, initialize=lambda m, s, t: \
        [r for r in m.R if data['sout'][r] == s and data['dout'][r] == t])
    m.R_use = pyo.Set(m.T, within=m.R, initialize=lambda m, t: \
        [r for r in m.R if data['dout'][r] < t and data['din'][r] >= t])
    m.R_not_g = pyo.Set(m.G, within=m.R, initialize=lambda m, g: \
        [r for r in m.R if data['gr'][r] != g])

    # --- 2.2. Parâmetros (Params) [cite: 195-233] ---
    m.PYU = pyo.Param(initialize=data['PYU'])
    m.BUD = pyo.Param(initialize=data['BUD'])
    m.M = pyo.Param(initialize=1e9)
    m.gr = pyo.Param(m.R, initialize=data['gr'])
    m.DEM = pyo.Param(m.R, m.A, m.P, initialize=data['DEM'], default=0)
    m.LEA = pyo.Param(m.G, initialize=data['LEA'])
    m.LP = pyo.Param(m.G, initialize=data['LP'])
    m.COS = pyo.Param(m.G, initialize=data['COS'])
    m.OWN = pyo.Param(m.G, initialize=data['OWN'])
    m.PRI = pyo.Param(m.P, m.G, initialize=data['PRI'])
    m.UPG = pyo.Param(m.G, m.G, initialize=data['UPG'])
    m.TC = pyo.Param(m.G, m.S, m.S, initialize=data['TC'])
    m.TT = pyo.Param(m.S, m.S, initialize=data['TT'])
    m.INX_O = pyo.Param(m.G, m.S, initialize=data['INX_O'])
    m.ONY_L = pyo.Param(m.G, m.T, m.S, initialize=data['ONY_L'])
    m.ONY_O = pyo.Param(m.G, m.T, m.S, initialize=data['ONY_O'])
    m.ONU_L = pyo.Param(m.G, m.T, m.S, initialize=data['ONU_L'])
    m.ONU_O = pyo.Param(m.G, m.T, m.S, initialize=data['ONU_O'])
    
    # Parâmetros de Preço Fixo
    m.p_chosen = pyo.Param(m.R, m.A, initialize=price_strategy)
    
    def dem_fixed_init(m, r, a):
        p_star = m.p_chosen[r,a]
        # Adicionar verificação para chaves em falta
        if (r, a, p_star) not in m.DEM:
             print(f"Aviso: Chave DEM em falta ({r}, {a}, {p_star}). A usar 0.")
             return 0
        return m.DEM[r,a,p_star]
    m.DEM_fixed = pyo.Param(m.R, m.A, initialize=dem_fixed_init)
    
    def pri_fixed_init(m, r, a):
        p_star = m.p_chosen[r,a]
        g_req = m.gr[r]
        if (p_star, g_req) not in m.PRI:
            print(f"Aviso: Chave PRI em falta ({p_star}, {g_req}). A usar 0.")
            return 0
        return m.PRI[p_star, g_req]
    m.PRI_fixed = pyo.Param(m.R, m.A, initialize=pri_fixed_init)

    # --- 2.3. Variáveis de Decisão (Variables) [cite: 235-248] ---
    m.w_O = pyo.Var(m.G, m.S, domain=pyo.NonNegativeIntegers)
    m.w_L = pyo.Var(m.G, m.T, m.S, domain=pyo.NonNegativeIntegers)
    m.x = pyo.Var(m.G, m.T, m.S, m.FleetType, domain=pyo.NonNegativeIntegers)
    m.y = pyo.Var(m.S, m.S, m.G, m.T, m.FleetType, domain=pyo.NonNegativeIntegers)
    m.u = pyo.Var(m.R, m.A, m.G, m.FleetType, domain=pyo.NonNegativeIntegers)
    m.f = pyo.Var(m.G, m.T, m.FleetType, domain=pyo.NonNegativeIntegers)

    print("Conjuntos, Parâmetros e Variáveis (MIP) definidos.")

    # --- 2.4. Função Objetivo (Objective Function) - LINEAR [cite: 261-262] ---
    def revenue_term_linear(m):
        expr = 0
        for r in m.R:
            for a in m.A:
                total_u_rag = sum(m.u[r,a,g,'L'] + m.u[r,a,g,'O'] for g in m.G)
                price_fixed = m.PRI_fixed[r,a]
                expr += total_u_rag * price_fixed
        return expr

    def buying_cost_term(m):
        return sum(m.w_O[g,s] * m.COS[g] for g in m.G for s in m.S)
    def leasing_cost_term(m):
        return sum(m.f[g,t,'L'] * m.LEA[g] for g in m.G for t in m.T if t > 0)
    def ownership_cost_term(m):
        return sum(m.f[g,t,'O'] * m.OWN[g] for g in m.G for t in m.T if t > 0)
    def transfer_cost_term(m):
        return sum((m.y[s1,s2,g,t,'L'] + m.y[s1,s2,g,t,'O']) * m.TC[g,s1,s2] 
                   for s1 in m.S for s2 in m.S for g in m.G for t in m.T if t > 0)
    def upgrade_penalty_term(m):
        return sum((m.u[r,a,g,'L'] + m.u[r,a,g,'O']) * m.PYU 
                   for g in m.G for r in m.R_not_g[g] for a in m.A)

    m.Objective = pyo.Objective(
        expr = revenue_term_linear(m) - buying_cost_term(m) - leasing_cost_term(m) - \
               ownership_cost_term(m) - transfer_cost_term(m) - upgrade_penalty_term(m),
        sense = pyo.maximize
    )
    
    print("Função Objetivo (MIP) definida.")

    # --- 2.5. Restrições (Constraints) [cite: 264-350] ---

    # Eq. 2: Stock Owned [cite: 270-272]
    def stock_owned_rule(m, g, s, t):
        if t == 0: return pyo.Constraint.Skip
        arrivals_rentals = sum(m.u[r,a,g,'O'] for r in m.R_in[s,t] for a in m.A)
        departures_rentals = sum(m.u[r,a,g,'O'] for r in m.R_out[s,t] for a in m.A)
        arrivals_transfers = 0
        for c in m.S:
            if c == s: continue
            transfer_time = m.TT[c,s]
            if t - transfer_time - 1 >= 0:
                arrivals_transfers += m.y[c,s,g, t - transfer_time - 1, 'O']
        departures_transfers = sum(m.y[s,c,g, t-1, 'O'] for c in m.S if c != s and t-1 >= 0)
        return m.x[g,t,s,'O'] == m.x[g,t-1,s,'O'] + \
                                m.ONY_O[g,t,s] + m.ONU_O[g,t,s] + \
                                arrivals_rentals - departures_rentals + \
                                arrivals_transfers - departures_transfers
    m.stock_owned_con = pyo.Constraint(m.G, m.S, m.T, rule=stock_owned_rule)

    # Eq. 3 & 4: Stock Leased [cite: 280-286]
    def stock_leased_rule(m, g, s, t):
        if t == 0: return pyo.Constraint.Skip
        arrivals_rentals = sum(m.u[r,a,g,'L'] for r in m.R_in[s,t] for a in m.A)
        departures_rentals = sum(m.u[r,a,g,'L'] for r in m.R_out[s,t] for a in m.A)
        arrivals_transfers = 0
        for c in m.S:
            if c == s: continue
            transfer_time = m.TT[c,s]
            if t - transfer_time - 1 >= 0:
                arrivals_transfers += m.y[c,s,g, t - transfer_time - 1, 'L']
        departures_transfers = sum(m.y[s,c,g, t-1, 'L'] for c in m.S if c != s and t-1 >= 0)
        acquisition = m.w_L[g, t-1, s] if t-1 >= 0 else 0
        
        if t <= m.LP[g]: # Eq 3
             return m.x[g,t,s,'L'] == m.x[g,t-1,s,'L'] + \
                                    m.ONY_L[g,t,s] + m.ONU_L[g,t,s] + \
                                    arrivals_rentals - departures_rentals + \
                                    arrivals_transfers - departures_transfers + \
                                    acquisition
        else: # Eq 4
            return_period = t - m.LP[g] - 1
            returned = m.w_L[g, return_period, s] if return_period >= 0 else 0
            return m.x[g,t,s,'L'] == m.x[g,t-1,s,'L'] + \
                                    m.ONY_L[g,t,s] + m.ONU_L[g,t,s] + \
                                    arrivals_rentals - departures_rentals + \
                                    arrivals_transfers - departures_transfers + \
                                    acquisition - returned
    m.stock_leased_con = pyo.Constraint(m.G, m.S, m.T, rule=stock_leased_rule)

    # Eq. 5: Initial Stock Owned
    def initial_stock_owned_rule(m, g, s):
        return m.x[g,0,s,'O'] == m.INX_O[g,s] + m.w_O[g,s]
    m.initial_stock_owned_con = pyo.Constraint(m.G, m.S, rule=initial_stock_owned_rule)

    # Eq. 6: Initial Stock Leased
    def initial_stock_leased_rule(m, g, s):
        return m.x[g,0,s,'L'] == 0
    m.initial_stock_leased_con = pyo.Constraint(m.G, m.S, rule=initial_stock_leased_rule)

    # Eq. 7: Capacity/Demand 1 (Stock Limit)
    def capacity_rule(m, g, s, t, ft):
        if t == 0: return pyo.Constraint.Skip
        rentals_out = sum(m.u[r,a,g,ft] for r in m.R_out[s,t] for a in m.A)
        transfers_out = sum(m.y[s,c,g,t,ft] for c in m.S if c != s)
        return rentals_out + transfers_out <= m.x[g,t,s,ft]
    m.capacity_con = pyo.Constraint(m.G, m.S, m.T, m.FleetType, rule=capacity_rule)

    # Eq. 8: Capacity/Demand 2 (Demand Limit) - LINEARIZED
    def demand_rule_linear(m, r, a):
        rentals_satisfied = sum(m.u[r,a,g,'L'] + m.u[r,a,g,'O'] for g in m.G)
        return rentals_satisfied <= m.DEM_fixed[r,a]
    m.demand_con = pyo.Constraint(m.R, m.A, rule=demand_rule_linear)

    # Eq. 9: Business-related (Upgrades)
    def upgrade_rule(m, r, g):
        rentals_satisfied_with_g = sum(m.u[r,a,g,'L'] + m.u[r,a,g,'O'] for a in m.A)
        return rentals_satisfied_with_g <= m.UPG[m.gr[r], g] * m.M
    m.upgrade_con = pyo.Constraint(m.R, m.G, rule=upgrade_rule)

    # Eq. 10: Business-related (Budget)
    def budget_rule(m):
        return sum(m.w_O[g,s] * m.COS[g] for g in m.G for s in m.S) <= m.BUD
    m.budget_con = pyo.Constraint(rule=budget_rule)

    # Eq. 12: Other (Auxiliary Fleet)
    def aux_fleet_rule(m, g, t, ft):
        if t == 0:
            return m.f[g,t,ft] == sum(m.x[g,t,s,ft] for s in m.S)
        stock_at_locations = sum(m.x[g,t,s,ft] for s in m.S)
        rentals_in_use = sum(m.u[r,a,g,ft] for r in m.R_use[t] for a in m.A)
        transfers_in_transit = 0
        for s1 in m.S:
            for s2 in m.S:
                if s1 == s2:
                    continue
                # Verificar se (s1,s2) existe no TT
                TT_s1s2 = m.TT[s1,s2] if (s1,s2) in m.TT else 0
                if TT_s1s2 > 0:
                    for tau in range(max(0, t - int(TT_s1s2)), t):
                        transfers_in_transit += m.y[s1,s2,g,tau,ft]
        return m.f[g,t,ft] == stock_at_locations + rentals_in_use + transfers_in_transit
    m.aux_fleet_con = pyo.Constraint(m.G, m.T, m.FleetType, rule=aux_fleet_rule)


    print("Restrições (MIP) definidas. Modelo construído com sucesso.")
    return m

# --- 3. BLOCO DE EXECUÇÃO (LINEARIZADO) ---

if __name__ == "__main__":
    
    # --- Passo 1: Definir a pasta dos dados ---
    # Assumindo que os ficheiros CSV estão na mesma pasta que este script
    DATA_PATH = "." 

    # --- Passo 2: Carregar os dados ---
    data_completa = load_data_from_inst_excel(os.path.join(DATA_PATH, "Inst2.xlsx"))
    
    if data_completa is None:
        print("Falha ao carregar os dados. A sair.")
        exit()
        
    # --- Passo 3: Definir uma Estratégia de Preços (Exemplo) ---
    # O teu algoritmo genético (BRKGA) [cite: 370] iria gerar estas estratégias.
    print("A usar estratégia de preços fictícia (p=1 para tudo)...")
    strategy = {}
    R_set = range(1, data_completa['R'] + 1)
    A_set = range(0, data_completa['A'] + 1)
    
    # Garantir que P_max é um inteiro válido
    P_max = data_completa.get('P', 1)
    if P_max < 1: P_max = 1
    
    for r in R_set:
        for a in A_set:
            # Usar sempre o nível de preço 1 (o mais baixo)
            strategy[(r,a)] = 1 
            # Verificação de segurança: se p=1 não existir, usar p=P_max
            if (r,a,1) not in data_completa['DEM']:
                strategy[(r,a)] = P_max
            
    # --- Passo 4: Criar o modelo linearizado ---
    model = create_linearized_model(data_completa, strategy)

    # --- Passo 5: Resolver o modelo (com solver MIP gratuito) ---
    print("\n--- A INICIAR O SOLVER (MIP) ---")
    
    solver_name = 'cbc'  # Pode usar 'glpk', 'cbc', 'gurobi', 'cplex'
    cbc_path = r"C:\Users\daniel.f.pereira\Anaconda3\Library\bin\cbc.exe"

    # Verifica se o executável existe
    if not os.path.exists(cbc_path):
        print(f"❌ Erro: CBC não encontrado em {cbc_path}")
        print("Verifica o caminho e tenta novamente.")
        exit()

    try:
        solver = pyo.SolverFactory('cbc', executable=cbc_path)
        if not solver.available(exception_flag=False):
            raise RuntimeError("Pyomo não conseguiu inicializar o CBC.")
        print(f"✅ Solver CBC encontrado em: {cbc_path}")
    except Exception as e:
        print(f"\nErro Crítico: não foi possível inicializar o solver CBC.")
        print("Detalhe:", e)
        exit()

    # --- Limite de tempo ---
    time_limit_segundos = 300  # 5 minutos
    solver.options['sec'] = time_limit_segundos

        
    print(f"A chamar o solver '{solver_name}' com um limite de tempo de {time_limit_segundos} segundos...")
    
    try:
        solver.options['log'] = 1
        results = solver.solve(model, tee=True)
    except Exception as e:
        print(f"ERRO DURANTE A RESOLUÇÃO: {e}")
        print("Isto pode indicar um problema nos dados (ex: chaves em falta) ou no solver.")
        exit()
    
    # --- Passo 6: Mostrar resultados ---
    print("\n--- RESULTADOS DA OTIMIZAÇÃO (MIP) ---")
    print(f"Condição de Terminação: {results.solver.termination_condition}")
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal or \
       (hasattr(model, 'Objective') and pyo.value(model.Objective, exception=False) is not None):
        print(f"Valor da Função Objetivo (Lucro): {pyo.value(model.Objective)}")
        
        print("\n--- Veículos 'Owned' a Comprar (w_O) ---")
        for g in model.G:
            for s in model.S:
                if pyo.value(model.w_O[g,s]) > 0.1:
                    print(f"  Grupo {g}, Local {s}: {pyo.value(model.w_O[g,s]):.0f} veículos")

        print("\n--- Alugueres Satisfeitos (u) (Exemplo para r=1) ---")
        if 1 in model.R:
            for a in model.A:
                for g in model.G:
                    u_L = pyo.value(model.u[1,a,g,'L'])
                    u_O = pyo.value(model.u[1,a,g,'O'])
                    if (u_L + u_O) > 0.1:
                        print(f"  Aluguer r=1, Antec. a={a}, Grupo {g}: {u_L+u_O:.0f} (L: {u_L:.0f}, O: {u_O:.0f})")
                        
    elif results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("O solver atingiu o limite de tempo.")
        if results.problem.number_of_solutions > 0 and hasattr(model, 'Objective'):
             print(f"Melhor solução (sub-ótima) encontrada: {pyo.value(model.Objective)}")
        else:
            print("Nenhuma solução viável encontrada dentro do limite de tempo.")
    else:
        print("O solver falhou ou não encontrou solução.")