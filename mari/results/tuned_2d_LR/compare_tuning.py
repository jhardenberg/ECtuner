#Argomenti da terminale:
#Per vedere solo la tabella: python compare_tuning.py -c config_tuner_2d.yaml
#Per vedere tabella + grafico: python compare_tuning.py -c config_tuner_2d.yaml --plot

#python compare_tuning.py -c ../../../config_tuner_2d.yaml -d yaml_files --plot -v (variables)
#python compare_tuning.py -c config.yaml -i a000 a050 a100 --plot -v (variables)
#python compare_tuning.py -c config.yaml -i tuned_phis_a030.yml tuned_phis_a095.yml --plot -v (variables)

import os, yaml, re, argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser(description='Comparison between different tuning strategies.')
    parser.add_argument('-d', '--dir', type=str, default='net_TOA/new_yaml_files', help='Results directory.')
    parser.add_argument('-c', '--config', type=str, help='Path to config file for OIFS reference.')
    parser.add_argument('--plot', action='store_true', help='Activate the plots.')
    parser.add_argument('-i', '--include', nargs='+', help='List of filenames to include OR keywords (e.g. a000 a100).')
    parser.add_argument('-v', '--vars', nargs='+', default=['net_toa'], help='Variables for tradeoff plot.')
    return parser.parse_args()

def extract_alpha(filename):
    """
    Gestisce i nomi file: a0 -> 0.0, a05 -> 0.5, a09 -> 0.9, a1 -> 1.0
    """
    match = re.search(r'_a(\d+)', filename)
    if not match: return None
    val_str = match.group(1)
    if val_str == "0": return 0.0
    if val_str == "1": return 1.0
    # Gestione formati tipo a05 (0.5) o a095 (0.95)
    if val_str.startswith('0'):
        return float(f"0.{val_str[1:]}")
    return float(val_str) / 10.0 if len(val_str) == 1 else float(val_str) / 100.0

def load_results(results_dir, config_path=None, include_list=None):
    all_data = []
    param_names = []
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
            ref = conf.get('reference_parameters', {})
            ref_row = {p: float(v) for p, v in ref.items()}
            ref_row['Experiment'] = 'REFERENCE_OIFS'
            ref_row['Alpha'] = -0.1 
            all_data.append(ref_row)
            param_names = list(ref.keys())

    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found.")
        return None, []

    files = [f for f in os.listdir(results_dir) if f.endswith((".yml", ".yaml"))]

    # --- LOGICA DI FILTRO ---
    if include_list:
        files = [f for f in files if any(key in f for key in include_list)]
    # ------------------------

    for filename in files:
        path = os.path.join(results_dir, filename)
        with open(path, 'r') as f:
            raw_text = f.read()
            f.seek(0)
            try:
                content = yaml.safe_load(f)
                params = {
                    'Experiment': filename.replace('.yml', '').replace('.yaml', ''),
                    'Alpha': extract_alpha(filename)
                }
                # Caricamento parametri
                tuning_block = content[0]['base.context']['model_config']['oifs']['tuning']
                for group in tuning_block.values():
                    params.update({p: float(v) for p, v in group.items()})
                
                m_spat = re.search(r'# total_spatial_cost: ([\d\.-]+)', raw_text)
                m_glob = re.search(r'# total_global_cost: ([\d\.-]+)', raw_text)
                m_metr = re.search(r'# metric_used: (\w+)', raw_text)
                
                metric = m_metr.group(1).lower() if m_metr else 'l2'
                params['Metric'] = metric

                # --- CONVERSIONE FISICA GLOBALE ---
                spat_tot = float(m_spat.group(1)) if m_spat else 0
                glob_tot = float(m_glob.group(1)) if m_glob else 0

                params['Raw_Spatial_Cost'] = spat_tot
                params['Raw_Global_Cost'] = glob_tot
                # somma pesata che l'ottimizzatore cerca di minimizzare (no penalità sui parametri)
                params['Total_Objective_Score'] = spat_tot + glob_tot
                
                # Creiamo metriche fisiche globali (per il Grafico 2)
                if metric == 'l2':
                    params['Phys_Spatial_Total'] = np.sqrt(spat_tot)
                    params['Phys_Global_Total'] = np.sqrt(glob_tot)
                    params['Phys_Label'] = "RMSE & Abs Bias"
                else:
                    params['Phys_Spatial_Total'] = spat_tot
                    params['Phys_Global_Total'] = glob_tot
                    params['Phys_Label'] = "MAE & Abs Bias"

                # Creiamo metriche fisiche per singola variabile (per il Grafico 3)
                for var in ['net_toa', 'rsnt', 'rlnt', 'swcf', 'lwcf']:
                    bias_m = re.search(fr'# {var}_global_bias_final: ([\d\.-]+)', raw_text)
                    cost_m = re.search(fr'# {var}_spatial_cost_(?:\w+): ([\d\.-]+)', raw_text)
                    
                    if bias_m and cost_m:
                        b_val = float(bias_m.group(1))
                        c_val = float(cost_m.group(1))
                        params[f'{var}_AbsBias'] = abs(b_val)
                        params[f'{var}_PhysSpatial'] = np.sqrt(c_val) if metric == 'l2' else c_val
                
                all_data.append(params)
            except Exception: continue
    
    df = pd.DataFrame(all_data).set_index('Experiment')
    return df, param_names

def plot_parameter_evolution(df, param_names, metric_filter=None):
    """% parameter evolution"""
    if metric_filter:
        df = df[df['Metric_Type'] == metric_filter]
    
    plt.figure(figsize=(12, 7))
    exp_df = df[df.index != 'REFERENCE_OIFS'].dropna(subset=['Alpha']).sort_values('Alpha')
    # Prendi solo la prima riga di reference per evitare duplicati L1/L2
    ref_values = df[df.index == 'REFERENCE_OIFS'].iloc[0]
    
    colormap = plt.get_cmap('tab20')
    for i, p in enumerate(param_names):
        if p in exp_df.columns:
            denom = ref_values[p] if ref_values[p] != 0 else 1e-15
            rel_change = ((exp_df[p] - ref_values[p]) / denom) * 100
            plt.plot(exp_df['Alpha'], rel_change, label=p, marker='o', color=colormap(i/len(param_names)))

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Parameter Evolution (%) {'- ' + metric_filter if metric_filter else ''}")
    plt.xlabel("Alpha")
    plt.ylabel("Variation (%) relative to OIFS")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

def plot_tradeoff_comparison(df):
    """Grafico Pareto Front Globale: L1 vs L2 con simboli diversi e scala Viridis"""
    plt.figure(figsize=(12, 8))
    
    # Mappa dei simboli per distinguere le metriche
    marker_map = {
        'L1 (Linear)': 'o',      # Cerchio
        'L2 (Quadratic)': 's'    # Quadrato
    }
    
    # Per gestire la colorbar unica
    last_sc = None

    # Ciclo sulle metriche presenti (L1, L2)
    for m_type in df['Metric_Type'].unique():
        # Filtriamo i dati validi per la metrica corrente
        subset = df[df['Metric_Type'] == m_type].dropna(subset=['Phys_Spatial_Total', 'Phys_Global_Total']).sort_values('Alpha')
        
        if subset.empty:
            continue
            
        mkr = marker_map.get(m_type, 'p') # 'p' come fallback
        
        # Disegniamo i punti colorati in base ad Alpha
        last_sc = plt.scatter(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], 
                              c=subset['Alpha'], cmap='viridis', 
                              marker=mkr, s=180, edgecolors='black', 
                              zorder=3, vmin=0, vmax=1)
        
        # Disegniamo la linea di trend tratteggiata
        plt.plot(subset['Phys_Spatial_Total'], subset['Phys_Global_Total'], 
                 linestyle='--', alpha=0.3, zorder=2)

        # Annotazioni Alpha con simbolo LaTeX
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', 
                         (row['Phys_Spatial_Total'], row['Phys_Global_Total']),
                         xytext=(7, 7), textcoords='offset points', 
                         fontsize=8, alpha=0.8)

    # --- LEGENDA MANUALE PER I SIMBOLI ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='L1 (Linear)',
               markerfacecolor='gray', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='L2 (Quadratic)',
               markerfacecolor='gray', markersize=12, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Metrica Obiettivo")

    # --- CONFIGURAZIONE ESTETICA ---
    plt.title("Pareto Front: Total Spatial Error vs Total Global Bias (all_fluxes)", fontsize=14)
    plt.xlabel("Total Spatial Error (Physical Units)", fontsize=12)
    plt.ylabel("Total Global Bias (Physical Units)", fontsize=12)
    
    # Aggiunta della Colorbar per Alpha
    if last_sc:
        cbar = plt.colorbar(last_sc)
        cbar.set_label(r'Alpha Value ($\alpha$)', fontsize=12)
    
    plt.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.tight_layout()

def plot_variable_pareto(df, var):
    """Grafico Pareto Front: L1 vs L2 con simboli diversi e scala Viridis"""
    plt.figure(figsize=(12, 8))
    
    marker_map = {
        'L1 (Linear)': 'o',      # Cerchio
        'L2 (Quadratic)': 's'    # Quadrato
    }
    
    col_spat = f'{var}_PhysSpatial'
    col_bias = f'{var}_AbsBias'
    
    # Per gestire la colorbar unica, memorizziamo l'ultimo scatter creato
    last_sc = None

    # Ciclo sulle metriche presenti nel DataFrame
    for m_type in df['Metric_Type'].unique():
        subset = df[df['Metric_Type'] == m_type].dropna(subset=[col_spat, col_bias]).sort_values('Alpha')
        
        if subset.empty:
            continue
            
        # Simbolo specifico per questa metrica
        mkr = marker_map.get(m_type, 'p') # 'p' come fallback
        
        # Disegniamo i punti
        last_sc = plt.scatter(subset[col_spat], subset[col_bias], 
                              c=subset['Alpha'], cmap='viridis', 
                              marker=mkr, s=180, edgecolors='black', 
                              label=mkr, # Temporaneo per la legenda dei simboli
                              zorder=3, vmin=0, vmax=1)
        
        # Disegniamo la linea di trend (opzionale, molto sottile)
        plt.plot(subset[col_spat], subset[col_bias], 
                 linestyle='--', alpha=0.3, zorder=2)

        # Annotazioni Alpha con simbolo LaTeX
        for _, row in subset.iterrows():
            plt.annotate(fr'$\alpha$={row["Alpha"]:.2f}', 
                         (row[col_spat], row[col_bias]),
                         xytext=(7, 7), textcoords='offset points', 
                         fontsize=8, alpha=0.8)

    # --- LEGENDA MANUALE PER I SIMBOLI ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='L1 (Linear)',
               markerfacecolor='gray', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='L2 (Quadratic)',
               markerfacecolor='gray', markersize=12, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Metrics Type")

    # --- CONFIGURAZIONE ESTETICA ---
    plt.title(f"Pareto Front Comparison: Spatial Error vs Global Bias (all_fluxes, {var})", fontsize=14)
    plt.xlabel(f"Spatial Error (Physical Units)", fontsize=12)
    plt.ylabel(f"Absolute Global Bias (Physical Units)", fontsize=12)
    
    # Colorbar per Alpha
    if last_sc:
        cbar = plt.colorbar(last_sc)
        cbar.set_label(r'Alpha Value ($\alpha$)', fontsize=12)
    
    plt.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.tight_layout()

# --- FLUSSO TERMINALE ---

def main():
    # Se lanciato da terminale, usa backend Agg per non crashare sui cluster
    matplotlib.use('Agg')
    args = parse_args()
    df, param_names = load_results(args.dir, args.config, args.include)

    if df is None or df.empty: return

    # Aggiungi colonna fittizia se non caricata dal notebook
    if 'Metric_Type' not in df.columns:
        df['Metric_Type'] = df['Metric'].map({'l1': 'L1 (Linear)', 'l2': 'L2 (Quadratic)'})

    df = df.sort_values(by='Alpha')

    if args.plot:
        # Salvataggio automatico file PNG
        plot_parameter_evolution(df, param_names)
        plt.savefig(os.path.join(args.dir, 'parameter_evolution_sweep.png'))
        
        plot_tradeoff_comparison(df)
        plt.savefig(os.path.join(args.dir, 'performance_tradeoff_sweep.png'))
        
        for v in args.vars:
            plot_variable_pareto(df, v)
            plt.savefig(os.path.join(args.dir, f'physical_tradeoff_{v}.png'))
        
        print(f"Plots saved in: {args.dir}")

if __name__ == "__main__":
    main()