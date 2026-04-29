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
# Obbligatorio per lavorare su cluster senza display
matplotlib.use('Agg')

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
    match = re.search(r'_a(\d{3})', filename) # Cerca esattamente 3 cifre dopo '_a'
    if match:
        return float(match.group(1)) / 100.0
    
    # Fallback per il vecchio formato (es: _a0, _a05, _a1)
    match_old = re.search(r'_a(\d+)', filename)
    if not match_old: return None
    val_str = match_old.group(1)
    if len(val_str) > 1 and val_str.startswith('0'):
        return float(f"0.{val_str[1:]}")
    val = float(val_str)
    return val / 10.0 if val > 1 else val

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
                    cost_m = re.search(fr'# {var}_(?:spatial_cost_final|rmse_spat_final): ([\d\.-]+)', raw_text)
                    
                    if bias_m and cost_m:
                        b_val = float(bias_m.group(1))
                        c_val = float(cost_m.group(1))
                        params[f'{var}_AbsBias'] = abs(b_val)
                        # Se L2, convertiamo il costo spaziale in RMSE (radice)
                        params[f'{var}_PhysSpatial'] = np.sqrt(c_val) if metric == 'l2' else c_val
                
                all_data.append(params)
            except Exception: continue
    
    df = pd.DataFrame(all_data).set_index('Experiment')
    return df, param_names

def main():
    args = parse_args()
    df, param_names = load_results(args.dir, args.config, args.include)

    if df is None or df.empty:
        print("Error: no data found in specified paths")
        return

    df = df.sort_values(by='Alpha')

    # --- SALVATAGGIO TABELLA ---
    suffix = "_filtered" if args.include else ""
    table_path = os.path.join(args.dir, f'summary_table_comparison{suffix}.txt')
    df_display = df.dropna(axis=1, how='all').drop(columns=['Alpha'], errors='ignore')
    table_output = tabulate(df_display, headers='keys', tablefmt='psql', floatfmt=".3e")
    
    with open(table_path, 'w') as f:
        f.write(table_output)
        f.write("\n\nNote: All spatial errors are converted to physical units (RMSE for L2, MAE for L1).\n")
    
    print(f"Done! Table saved in: {table_path}")

    if args.plot:
        # Prepariamo i percorsi per i grafici
        plot_param_path = os.path.join(args.dir, 'parameter_evolution_sweep.png')
        plot_perf_path = os.path.join(args.dir, 'performance_tradeoff_sweep.png')

        # --- GRAFICO 1: EVOLUZIONE PARAMETRI ---
        if 'REFERENCE_OIFS' in df.index:
            plt.figure(figsize=(12, 7))
            exp_df = df.drop('REFERENCE_OIFS').dropna(subset=['Alpha'])
            ref_values = df.loc['REFERENCE_OIFS', param_names]
            
            colormap = plt.get_cmap('tab20') 
            num_params = len(param_names)

            for i, p in enumerate(param_names):
                if p in exp_df.columns:
                    denom = ref_values[p] if ref_values[p] != 0 else 1e-15
                    rel_change = ((exp_df[p] - ref_values[p]) / denom) * 100
                    
                    # Assegniamo un colore unico basato sull'indice i
                    color = colormap(i / num_params) if num_params > 10 else None 
                    
                    plt.plot(exp_df['Alpha'], rel_change, label=p, marker='o', alpha=0.8, color=color)

            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.title("Parameter evolution (%) across Alpha sweep")
            plt.xlabel("Alpha (0=Spatial, 1=Global)")
            plt.ylabel("Variation (%) relative to OIFS")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(plot_param_path, dpi=150)

            # --- GRAFICO 2: TRADE-OFF DEI COSTI TOTALI (TUTTE LE VARIABILI) ---
        if 'Phys_Spatial_Total' in df.columns:
            plt.figure(figsize=(10, 7))
            perf_df = df.dropna(subset=['Phys_Spatial_Total', 'Phys_Global_Total'])
            metric_label = perf_df['Phys_Label'].iloc[0]

            sc = plt.scatter(perf_df['Phys_Spatial_Total'], perf_df['Phys_Global_Total'], 
                             c=perf_df['Alpha'], cmap='coolwarm', s=200, edgecolors='black', zorder=3)
            
            plt.plot(perf_df['Phys_Spatial_Total'], perf_df['Phys_Global_Total'], linestyle='--', color='gray', alpha=0.5)

            for idx, row in perf_df.iterrows():
                plt.annotate(f"a={row['Alpha']:.2f}", (row['Phys_Spatial_Total'], row['Phys_Global_Total']), 
                             xytext=(8,8), textcoords='offset points', fontsize=9)

            plt.title(f"Global Physical Trade-off ({metric_label})")
            plt.xlabel("Weighted Spatial Error ($W/m^2$)")
            plt.ylabel("Weighted Global Bias ($W/m^2$)")
            plt.colorbar(sc, label='Alpha')
            plt.grid(True, alpha=0.2)
            plt.savefig(os.path.join(args.dir, 'total_physical_tradeoff.png'), dpi=150)
            plt.close()
        
        # --- GRAFICO 3: PERFORMANCE ---
        for var in args.vars:
            phys_spat_col = f'{var}_PhysSpatial'
            abs_bias_col = f'{var}_AbsBias'

            if phys_spat_col in df.columns and abs_bias_col in df.columns:
                plt.figure(figsize=(10, 6))
                metric_used = df['Metric'].dropna().iloc[0].upper()
                x_label = "RMSE" if metric_used == 'L2' else "MAE"

                sc = plt.scatter(df[phys_spat_col], df[abs_bias_col], 
                                 c=df['Alpha'], cmap='viridis', s=150, edgecolors='black')
                
                for idx, row in df.iterrows():
                    if not np.isnan(row[phys_spat_col]):
                        plt.annotate(f"a={row['Alpha']:.2f}", (row[phys_spat_col], row[abs_bias_col]), 
                                     xytext=(5,5), textcoords='offset points', fontsize=8)

                plt.colorbar(sc, label='Alpha')
                plt.title(f"Physical Pareto Front: {var} ({metric_used})")
                plt.xlabel(f"Spatial {x_label} ($W/m^2$)")
                plt.ylabel(f"Absolute Global Bias ($W/m^2$)")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(args.dir, f'physical_tradeoff_{var}.png'), dpi=150)
                plt.close()

        print(f"Plots saved in: {args.dir}")

if __name__ == "__main__":
    main() 