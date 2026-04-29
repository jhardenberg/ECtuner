#!/usr/bin/env python3
"""
ectuner_diag.py
Diagnostic tool: dato reference, slope, sensitivities e i valori base (net_toa, tas)
calcola ΔT, target corretto e stima via least-squares delle variazioni parametriche richieste.
"""

import yaml
import numpy as np
import argparse
import sys

def load_yaml(p): 
    with open(p, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='gm_reference YAML')
    p.add_argument('--slope', required=True, help='slope YAML')
    p.add_argument('--sens', required=True, help='sensitivity YAML')
    p.add_argument('--nettoa', type=float, required=True, help='net_toa model (W/m2)')
    p.add_argument('--tas', type=float, required=True, help='tas model (°C)')
    p.add_argument('--tasref', type=float, help='tas reference override (°C, optional)')
    p.add_argument('--inc', type=float, default=0.2, help='inc (fractional allowed change wrt reference pars)')
    p.add_argument('--refpars', required=False, help='yaml file with reference_pars (optional, needed to compute absolute bounds)')
    args = p.parse_args()

    ref = load_yaml(args.ref)
    slopes = load_yaml(args.slope)
    sens = load_yaml(args.sens)

    # read reference values (ALL Global)
    try:
        net_ref = ref['net_toa']['obs']['ALL']['Global']['mean']
        if args.tasref is not None:
            tas_ref = args.tasref
        else:
            tas_ref = ref['tas']['obs']['ALL']['Global']['mean']
    except Exception as e:
        print("Errore lettura reference (controlla le chiavi).", e)
        sys.exit(1)

    net_model = args.nettoa
    tas_model = args.tas

    deltaT = tas_model - tas_ref
    print(f"\nReference net_toa = {net_ref:.6f} W/m2, tas_ref = {tas_ref:.6f} °C")
    print(f"Model: net_toa = {net_model:.6f} W/m2, tas = {tas_model:.6f} °C")
    print(f"ΔT = tas_model - tas_ref = {deltaT:.6f} K")

    # slope for net_toa (ALL Global)
    try:
        slope_net = slopes['T_slope']['net_toa']['ALL']['Global']
    except Exception as e:
        print("Errore lettura slope net_toa:", e)
        sys.exit(1)
    print(f"slope(net_toa) = {slope_net:.6f} W/m2/K")

    # corrected reference
    net_ref_corr = net_ref - (deltaT * slope_net)
    deltaF = net_ref_corr - net_model
    print(f"Corrected net_toa reference = {net_ref_corr:.6f} W/m2")
    print(f"ΔF required = net_ref_corr - net_model = {deltaF:.6f} W/m2\n")

    # Build sensitivity row for net_toa over parameters in sens
    params = sorted(list(sens.keys()))
    S_row = []
    for pnm in params:
        v = 0.0
        try:
            v = sens[pnm]['net_toa']['ALL']['Global'][0]
        except Exception:
            v = 0.0
        S_row.append(v)
    S = np.array(S_row).reshape(1, -1)  # 1 x N_params

    # Solve least-squares (min-norm) for delta p (S dp = ΔF)
    dF_vec = np.array([deltaF])
    dp_ls, residuals, rank, svals = np.linalg.lstsq(S, dF_vec, rcond=None)
    dp_ls = dp_ls.ravel()
    print("Params considered (count={}):".format(len(params)))
    for i, pnm in enumerate(params):
        print(f"  {i:2d} {pnm:20s}  S={S[0,i]:12.6e}  dp_ls={dp_ls[i]:12.6e}")

    # if refpars provided, compute bounds
    if args.refpars:
        refpars = load_yaml(args.refpars)
        print("\nBounds check (using provided reference_pars and inc):")
        inc = args.inc
        for i, pnm in enumerate(params):
            refv = refpars.get(pnm, None)
            if refv is None:
                print(f"  {pnm:20s}  (no refpar)")
                continue
            max_allowed = refv * inc
            if abs(dp_ls[i]) > max_allowed:
                print(f"  ⚠ {pnm:20s} dp_ls={dp_ls[i]:.6e} > allowed={max_allowed:.6e}")
            else:
                print(f"  OK {pnm:20s} dp_ls={dp_ls[i]:.6e} <= allowed={max_allowed:.6e}")
    else:
        print("\nNota: non hai fornito 'reference_pars' (file). Se li fornisci script segnalerà i bound tramite --refpars <yaml> e --inc <val>.")

    # compute predicted ΔF if we saturate bounds (if refpars provided)
    if args.refpars:
        refpars = load_yaml(args.refpars)
        dp_bounds = []
        for i, pnm in enumerate(params):
            refv = refpars.get(pnm, None)
            if refv is None:
                dp_bounds.append(0.0)
            else:
                direction = -np.sign(S[0,i]*deltaF)
                dpb = direction * (abs(refv)*args.inc)
                dp_bounds.append(dpb)
        dp_bounds = np.array(dp_bounds)
        dF_max = S.dot(dp_bounds.reshape(-1,1)).ravel()[0]
        print(f"\nMax achievable ΔF if saturating bounds (inc={args.inc}): {dF_max:.6e} W/m2")
        print("Se dF_max < ΔF, allora non è possibile raggiungere il target con i parametri/bounds attuali.")

if __name__ == '__main__':
    main()