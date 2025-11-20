import os
print("CWD =", os.getcwd())
import math
import pandas as pd
import numpy as np
import sys
print(sys.executable)

# ============================
# 1. Leitura das planilhas base
# ============================

def load_unit_params(xls: pd.ExcelFile):
    """
    Lê aba UnitParameters e devolve dict {nome_param: valor}
    Espera, por exemplo:
      G, S, A, T, PYU, BUD
    """
    up = pd.read_excel(xls, "UnitParameters")
    names = up.iloc[:, 0].tolist()
    vals = up.iloc[:, 1].tolist()
    unit = dict(zip(names, vals))
    return unit


def load_parameters_by_group(xls: pd.ExcelFile):
    """
    Lê aba ParametersByGroup (sem header) no formato:
      linha 0: LEA_g
      linha 1: LP_g
      linha 2: COS_g
      linha 3: OWN_g (se existir)
    """
    pbg = pd.read_excel(xls, "ParametersByGroup", header=None)
    data = pbg.iloc[:, 1:].to_numpy(dtype=float)
    G = data.shape[1]
    LEA = data[0]
    LP = data[1].astype(int)
    COS = data[2]
    OWN = data[3] if data.shape[0] > 3 else np.zeros(G)
    return G, LEA, LP, COS, OWN


def load_rental_types(xls: pd.ExcelFile):
    """
    Lê aba RentalTypes.
    Espera colunas (ou equivalentes):
      - 'gr'   : grupo pedido
      - 'sout' : estação de saída
      - 'dout' : período de saída
      - 'din'  : período de devolução
    """
    rt = pd.read_excel(xls, "RentalTypes")
    return rt


def load_demands(xls: pd.ExcelFile):
    """
    Lê aba Demand no formato das instâncias:
      - primeira linha = header (DEM_rap, p=1,a=0, p=1,a=1, ...)
      - demais linhas = números de demanda por RentalType e (p,a)

    Aproximação:
      demand_r[r] = MAX_{p,a} DEM_{r,a,p}
    """
    dem_raw = pd.read_excel(xls, "Demand", header=None)
    body = dem_raw.iloc[1:]
    numeric = body.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    demand_max = numeric.max(axis=1).to_numpy()
    return demand_max  # len = número de RentalTypes


def load_upgrades(xls: pd.ExcelFile, G: int):
    """
    Lê aba Upgrades e cria uma matriz UPG[g1, g2] (1..G, 1..G)
    onde:
      UPG[g1, g2] = 1 se grupo g2 pode atender demanda do grupo g1.

    Se o formato da aba for diferente, talvez precise ajustar isso.
    Aqui assumo algo estilo matriz com 1/0 a partir da segunda linha/coluna.
    """
    upg_raw = pd.read_excel(xls, "Upgrades", header=None)

    # Se estiver vazia ou só com header, assume só self-upgrade
    if upg_raw.shape[0] <= 1:
        UPG = np.eye(G+1, dtype=int)
        return UPG

    mat = upg_raw.iloc[1:, 1:1+G].to_numpy(dtype=float)
    nrows = mat.shape[0]

    # se tiver menos linhas que G, completa com identidade
    if nrows < G:
        extra = np.zeros((G - nrows, G))
        for i in range(G - nrows):
            extra[i, i + nrows] = 1
        mat = np.vstack([mat, extra])

    # garante diagonal = 1
    for g in range(G):
        mat[g, g] = 1.0

    UPG = np.zeros((G+1, G+1), dtype=int)
    for g1 in range(1, G+1):
        for g2 in range(1, G+1):
            val = mat[g1-1, g2-1]
            UPG[g1, g2] = 1 if val >= 0.5 else 0

    return UPG


# =========================================
# 2. Construir D[g, s, t] a partir dos dados
# =========================================

def compute_D_matrix(unit, G, rt, demand_r):
    """
    Constrói D[g, s, t] com demanda "em uso"
    - group = gr
    - station = sout
    - períodos de uso = dout..din-1 (limitado ao horizonte)
    """
    S = int(unit['S'])
    T = int(unit['T'])

    D = np.zeros((G+1, S+1, T), dtype=float)

    R = len(rt)
    if len(demand_r) < R:
        demand_r = np.pad(demand_r, (0, R-len(demand_r)), constant_values=0.0)

    for idx, row in rt.iterrows():
        dem = float(demand_r[idx])
        if dem <= 0:
            continue

        g = int(row['gr'])
        s = int(row['sout'])
        dout = int(row['dout'])
        din  = int(row['din'])

        first_t = max(dout, 0)
        last_t  = min(din,  int(unit['T'])-1)
        if last_t < first_t:
            continue

        for t in range(first_t, last_t+1):
            D[g, s, t] += dem

    return D


# ============================================
# 3. Derivar φ (fração de upgrade) via PYU
# ============================================

def compute_phi_from_pyu(PYU: float):
    """
    Deriva φ a partir de PYU (custo de upgrade):
      φ = 1 / (1 + PYU)

    PYU = 0  -> φ = 1   (upgrade liberado)
    PYU = 1  -> φ = 0.5
    PYU = 3  -> φ = 0.25
    """
    if PYU <= 0:
        return 1.0
    return 1.0 / (1.0 + PYU)


# =======================================================
# 4. Aplicar upgrades para reduzir demanda "exigida" do grupo
# =======================================================

def apply_upgrades_to_demand(D, UPG, phi, G, S, T):
    """
    D[g,s,t]: demanda bruta.
    UPG[g1,g2]: se g2 pode atender demanda de g1.
    phi: fração máxima da demanda de g que pode subir para grupos superiores.

    Retorna:
      D_own[g,s,t] : parte da demanda que ainda exige carro do grupo g.
      Freq[g,s]    : pico dessa demanda (para dimensionar capacidade).
    """
    D_own = D.copy()

    for g in range(1, G+1):
        for s in range(1, S+1):
            for t in range(T):
                base_dem = D[g, s, t]
                if base_dem <= 0:
                    continue

                has_superior = any(UPG[g, h] == 1 and h != g for h in range(1, G+1))
                if not has_superior:
                    continue

                upgradable = phi * base_dem
                # remove só a parte que estamos dispostos a cobrir via upgrade
                D_own[g, s, t] = max(0.0, base_dem - upgradable)

    Freq = np.zeros((G+1, S+1), dtype=float)
    for g in range(1, G+1):
        for s in range(1, S+1):
            Freq[g, s] = D_own[g, s, :].max()

    return D_own, Freq


# ====================================
# 5. Aplicar budget na frota própria
# ====================================

def apply_budget(unit, G, COS, w_owned):
    """
    Se o custo da frota própria exceder BUD, escala tudo
    proporcionalmente para caber no orçamento.
    """
    BUD = float(unit['BUD'])
    S = w_owned.shape[1] - 1

    cost = 0.0
    for g in range(1, G+1):
        for s in range(1, S+1):
            cost += w_owned[g, s] * COS[g-1]

    if cost <= BUD or cost == 0:
        return w_owned

    alpha = BUD / cost
    new_w = np.floor(w_owned * alpha).astype(int)
    return new_w


# ====================================
# 6. Heurística por instância (COM UPGRADE)
# ====================================

def heuristic_capacity_instance(path, gamma=0.7):
    """
    Heurística construtiva para UMA instância (InstX.xlsx):

    - Lê parâmetros, demandas, upgrades
    - Constrói D[g,s,t]
    - Aplica upgrades limitados por φ(PYU)
    - Define frota própria w_owned[g,s]
    - Define leasing w_leased[g,t,s]
    """
    xls = pd.ExcelFile(path)

    unit = load_unit_params(xls)
    G, LEA, LP, COS, OWN = load_parameters_by_group(xls)
    rt   = load_rental_types(xls)
    demr = load_demands(xls)
    UPG  = load_upgrades(xls, G)

    PYU  = float(unit.get("PYU", 0.0))
    phi  = compute_phi_from_pyu(PYU)

    S = int(unit['S'])
    T = int(unit['T'])

    # Demanda bruta
    D = compute_D_matrix(unit, G, rt, demr)

    # Aplica upgrades "parciais" limitados por phi
    D_own, Freq = apply_upgrades_to_demand(D, UPG, phi, G, S, T)

    # Frota própria = fração do pico
    w_owned = np.ceil(gamma * Freq).astype(int)
    w_owned = apply_budget(unit, G, COS, w_owned)

    # Leasing cobre o restante
    w_leased = np.zeros((G+1, T, S+1), dtype=int)
    active   = np.zeros((G+1, T, S+1), dtype=int)

    for g in range(1, G+1):
        Lg = int(LP[g-1])
        for s in range(1, S+1):
            for t in range(T):
                need = int(math.ceil(D_own[g, s, t] - w_owned[g, s]))
                if need <= 0:
                    continue
                already = active[g, t, s]
                deficit = max(0, need - already)
                if deficit > 0:
                    w_leased[g, t, s] += deficit
                    for tau in range(t, min(T, t+Lg)):
                        active[g, tau, s] += deficit

    return unit, G, S, T, Freq, w_owned, w_leased, PYU, phi


# =============================================
# 7. Loop para TODAS as instâncias na pasta "data"
# =============================================

def run_heuristic_for_all_instances(
    folder_path: str,
    output_excel: str,
    gamma: float = 0.7,
):
    """
    folder_path  : pasta onde estão Inst1.xlsx ... Inst40.xlsx (ex.: "data")
    output_excel : nome do arquivo de saída .xlsx
    gamma        : fração do pico usada na frota própria (0.5–0.8)
    """
    paths = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".xlsx") and f.startswith("Inst")
        ],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("Inst", ""))
    )

    summary_rows = []

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        for p in paths:
            name = os.path.basename(p).replace(".xlsx", "")
            print(f"Processando {name}...")

            unit, G, S, T, Freq, w_owned, w_leased, PYU, phi = heuristic_capacity_instance(
                p, gamma=gamma
            )

            # ----- aba de frota própria -----
            owned_records = []
            for g in range(1, G+1):
                for s in range(1, S+1):
                    owned_records.append({
                        "group": g,
                        "station": s,
                        "w_owned": int(w_owned[g, s]),
                        "Freq_max": float(Freq[g, s])
                    })
            df_owned = pd.DataFrame(owned_records)
            df_owned.to_excel(writer, sheet_name=f"{name}_owned", index=False)

            # ----- aba de leasing (DENSO: todos os períodos) -----
            leased_records = []
            for g in range(1, G+1):
                for t in range(T):
                    for s in range(1, S+1):
                        val = int(w_leased[g, t, s])
                        leased_records.append({
                            "group": g,
                            "time": t,
                            "station": s,
                            "w_leased": val
                        })

            df_leased = pd.DataFrame(
                leased_records,
                columns=["group", "time", "station", "w_leased"]
            )
            df_leased.to_excel(writer, sheet_name=f"{name}_leased", index=False)

            # ----- resumo -----
            total_owned = sum(
                int(w_owned[g, s]) for g in range(1, G+1) for s in range(1, S+1)
            )

            summary_rows.append({
                "instance": name,
                "G": G,
                "S": S,
                "T": T,
                "PYU": PYU,
                "phi_used": phi,
                "total_owned": total_owned,
                "budget": float(unit["BUD"])
            })

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)


if __name__ == "__main__":
    folder = "data"  # <- AQUI é o nome da pasta com os Excels
    output_excel = "heuristic_fleet_40_instances_with_upgrades.xlsx"
    run_heuristic_for_all_instances(folder, output_excel, gamma=0.7)
    print(f"Arquivo gerado em: {output_excel}")
