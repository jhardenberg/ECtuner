#!/usr/bin/env python3
"""
plot_gregory.py

Apre file NetCDF con xarray + dask,
calcola medie annuali pesate, applica running mean, e disegna il plot Gregory.
"""

import argparse
import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def main(exp, fil_pattern):
    # --- Apertura dataset con dask ---
    
    print("Opening files:", fil_pattern)
    ds = xr.open_mfdataset(fil_pattern, use_cftime=True)

    # rinomina time_counter -> time se necessario
    if "time_counter" in ds.dims or "time_counter" in ds.coords:
        ds = ds.rename({"time_counter": "time"})

    # prendi solo le variabili utili
    vars_needed = ["rsut", "rlut", "rsdt", "tas"]
    vars_present = [v for v in vars_needed if v in ds.variables]
    ds = ds[vars_present]

    # porta lat in memoria (evita problemi di chunking)
    ds = ds.assign_coords(lat=ds.lat.compute())

    all_lats = ds.lat.groupby('lat').mean()
    weights = np.cos(np.deg2rad(all_lats))

    gigimean = (
    ds.groupby('time.year').mean()
        .groupby('lat').mean()
        .weighted(weights).mean('lat')
    )
    # variabile derivata: net TOA
    gigimean = gigimean[["rsut", "rlut", "rsdt", "tas"]]
    gigimean = gigimean.assign(
        toa_net=gigimean.rsdt - gigimean.rlut - gigimean.rsut
    )

    # calcolo effettivo (dataset piccolo, serie temporale annuale)
    print("Computing final aggregated dataset (this may take a while)...")
    gigimean = gigimean.compute()

    # --- Prepara dati per il plot ---
    years = gigimean["year"].values
    tas = gigimean.tas
    toa = gigimean.toa_net

    running20_tas = tas.rolling(year=20, center=True).mean()
    running20_toa = toa.rolling(year=20, center=True).mean()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(tas, toa, linewidth=1.0, label=f"{exp} path (annual)")
    ax.plot(running20_tas, running20_toa, linewidth=2.0, linestyle="-", label="Running mean 20y")

    # marker iniziale/finale
    year_min = int(years.min())
    year_max = int(years.max())
    init_slice = slice(year_min, min(year_min + 9, year_max))
    final_slice = slice(max(year_min, year_max - 9), year_max)

    init_tas = tas.sel(year=init_slice).mean().item()
    init_toa = toa.sel(year=init_slice).mean().item()
    final_tas = tas.sel(year=final_slice).mean().item()
    final_toa = toa.sel(year=final_slice).mean().item()

    ax.scatter(init_tas, init_toa, s=250, edgecolors="black", facecolors="none",
               marker="o", linewidth=1.5, label="Initial decade mean")
    ax.scatter(final_tas, final_toa, s=250, edgecolors="black", facecolors="blue",
               marker="o", label="Final decade mean")

    # --- Bande di riferimento ---
    tas_clim = 281.6
    net_toa_clim = 0.89
    dx = dy = 0.3
    xmin, xmax = float(tas.min().item()), float(tas.max().item())
    ymin, ymax = float(toa.min().item()), float(toa.max().item())

    ax.fill_betweenx(np.linspace(ymin, ymax, 200), tas_clim - dx, tas_clim + dx,
                     color="grey", alpha=0.2)
    ax.fill_between(np.linspace(xmin, xmax, 200), net_toa_clim - dy, net_toa_clim + dy,
                    color="grey", alpha=0.2)

    ax.set_xlabel("GTAS (K)")
    ax.set_ylabel("net TOA (W/m$^2$)")

    x_text_tas = tas_clim + dx/2
    y_text_tas = ymin
    ax.text(x_text_tas, y_text_tas, f"TAS wrong: {tas_clim:.2f} K",
            va="bottom", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.6))
    
    x_text_toa = xmax
    y_text_toa = net_toa_clim - dy/2
    ax.text(x_text_toa, y_text_toa,
            f"Net_TOA ref target: {net_toa_clim:.2f} W/m²\n(gm_reference_EC23.yml global mean 2000–2020)",
            va="bottom", ha="right", fontsize=9, bbox=dict(facecolor="white", alpha=0.6))

    # Fit slope e linea target
    slope, intercept = np.polyfit(tas.values, toa.values, 1)
    x_line = np.linspace(xmin, xmax, 200)
    y_line = slope * (x_line - tas_clim) + net_toa_clim
    ax.plot(x_line, y_line, "k--", label="Target line (slope fit)")

    ax.legend()
    ax.grid(alpha=0.25)

    # --- Salva figura ---
    out_dir = os.getcwd()
    outname = os.path.join(out_dir, f"{exp}_gregory_plot.png")
    plt.savefig(outname, dpi=200, bbox_inches="tight")
    print("Saved plot to:", outname)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Gregory-style TOA vs TAS from EC-Earth files.")
    parser.add_argument("-e", "--exp", required=True, help="Experiment name (e.g. pi13)")
    parser.add_argument("-p", "--pattern", default=None, help="File glob pattern (overrides default path)")
    args = parser.parse_args()

    if args.pattern:
        pat = args.pattern
    else:
        pat = f"/ec/res4/scratch/itas/ece4/{args.exp}/output/oifs/{args.exp}_atm_cmip6_1m_*.nc"

    main(args.exp, pat)