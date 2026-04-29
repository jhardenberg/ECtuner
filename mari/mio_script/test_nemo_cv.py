# In this code I develop basin metrics to compare the ocean tuning simulations. I start by testing it on one experiment

from curses import window
from importlib.resources import files
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec # GRIDSPEC !
from scipy import stats
import scipy
import matplotlib as mpl
from matplotlib import colors
import argparse
import os

def global_mean(var,area, v_area):
    
    if(var.ndim == 2):
        return np.nansum(var*area, axis=(0,1))/v_area
    elif(var.ndim == 3):
        return np.nansum(var*area, axis=(1,2))/v_area
    else:
        #return np.nansum(var*area, axis=(2,3))/v_area
        return (var*area).sum(axis=(2,3))/v_area
    
def PlotProfileMask(var1,color,label, ocean_area, levels, basin, ax, window, mask, v_area):

    years = var1.year.values[-1]
    area_basin = getBasinMask(ocean_area, basin, mask)

    var1_mean = getBasinMask(var1.sel(year=slice(years-window,years)).mean(dim='year'),basin, mask)

    v_mask = var1_mean/var1_mean
    v_area = (v_mask*ocean_area).sum(axis=(1,2))

    var1_global = global_mean(var1_mean,area_basin, v_area)

    ax.plot(var1_global, -levels/1000, color = color, label=label, linewidth=2.5) # color=basin['color'], label = basin['label'])

def load_data(exp):
    path = '/ec/res4/scratch/itas/ece4/'+exp+'/output/nemo/'
    files = path + exp + '_oce_1m_T_*.nc'
    data = xr.open_mfdataset(files, chunks={"time_counter": 1}).groupby("time_counter.year").mean()

    return data

def getBasinMask(var, basin, mask):

    boxvar = var * mask[basin['label']] 
    #boxvar[boxvar == 0] = 'NaN'

    return boxvar

#def plotThetao(ctrl, test, ocean_area, levels, mask, basins, exp, v_area):
def plotThetao(test, ocean_area, levels, mask, basins, exp, v_area):
     
    power = 1/2  # o 1/1.5
    fwd = lambda y: np.sign(y) * (abs(y) ** power)
    inv = lambda y: np.sign(y) * (abs(y) ** (1/power))

    fig = plt.figure(1, figsize=(16,7), tight_layout=True)
    gs = gridspec.GridSpec(2,4, figure=fig)

    window= 30

    ax = fig.add_subplot(gs[0:2,0:2])
    #PlotProfileMask(ctrl, 'g', 'ctrl', ocean_area, levels, basins[0],ax, window, mask, v_area)
    PlotProfileMask(test, 'r', exp, ocean_area, levels,basins[0],ax, window, mask, v_area)

    ax.vlines(0,-6,0, color='k', linestyles='dashed', alpha=0.3)
    ax.set_title('(a) '+basins[0]['label'], fontsize=14)
    ax.set_xlabel(r'Potential temperature [K]', fontsize=13)
    ax.set_ylabel('Depth [km]', fontsize=13)
    ax.set_ylim(-6,0)
    #ax.set_xlim(-1.5, 6.5)
    #ax.set_xticks([-1,0,1,2,3,4,5,6])
    #ax.set_xticklabels(["-1",'0','1','2','3','4','5','6'], fontsize=11)
    leg = ax.legend(loc='lower right', fontsize=12)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    #ax.grid()
    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_yticks([-6,-5,-4,-3,-2,-1,-0.5,-0.1, 0])
    ax.set_yticklabels(['6','5','4','3','2','1','0.5', '0.1', '0'], fontsize=11)
    fig_label = ['(b) ', '(c) ', '(d) ', '(e) ']
    k=1
    for i in range(2):
            for j in range(2):
                    ax = fig.add_subplot(gs[i,2+j])
                    #PlotProfileMask(ctrl, 'g', 'ctrl', ocean_area, levels,basins[k],ax, window, mask, v_area)
                    PlotProfileMask(test, 'r', exp,ocean_area, levels, basins[k],ax, window, mask, v_area)

                    #ax.vlines(0,-6,0, color='k', linestyles='dashed', alpha=0.3)
                    ax.set_title(fig_label[k-1]+basins[k]['label'], fontsize=14)
                    ax.set_yscale('function', functions=(fwd, inv))
                    ax.set_ylim(-6,0)
                    
                    #ax.set_xlim(-2,12)
                    #ax.set_xticks([-1,0,1,2,3,4,5,6,7,8,9])
                    #ax.set_xticklabels(["-1",'0','1','2','3','4','5','6','7','8','9'], fontsize=11)

                    ax.set_yticks([-6,-5,-4,-3,-2,-1,-0.5,-0.1, 0])
                    ax.set_yticklabels(['6','5','4','3','2','1','0.5', '0.1', '0'], fontsize=11)
                    #ax.grid(alpha=0.5)

                    k+=1
                    if(i==1):
                            ax.set_xlabel(r'Potential temperature [K]', fontsize=13)
                    if(j==0):
                            ax.set_ylabel('Depth [km]', fontsize=13)


    #plt.suptitle('Anomalous vertical profile of temperature', fontsize=15)
    plt.savefig('/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/'+exp+'/'+exp+'_Temperature.pdf', bbox_inches='tight')

def plotPattern(ctrl, test, exp):
          
        title = ['(a) ~5 m', '(b) ~500 m', '(c) ~1000 m', '(d) ~3200 m']
        lev_index = [0,19, 21, 26]

        window = 30
        fig = plt.figure(2, figsize=(5,8), tight_layout=True)
        gs = gridspec.GridSpec(5,3, figure=fig, height_ratios=[1,1,1,1,0.1])
            
        for i in range(4):

      
            ref = np.nanmean(ctrl[-window:,lev_index[i],:,:], axis=0) 

            ax = fig.add_subplot(gs[i,1:3],projection=ccrs.PlateCarree())
            map1 = np.nanmean(test[-window:,lev_index[i],:,:], axis=0) 
            
            map = map1-ref

            t_stat, p_value = scipy.stats.ttest_ind(test[-window:,lev_index[i],:,:], ref, equal_var=False)
            map = np.ma.masked_where(p_value>0.05, map)
            
            newcmap = mpl.colormaps['RdBu_r']
            clevels = np.arange(-0.5, 0.6, 0.1)
            divnorm = colors.BoundaryNorm(clevels, ncolors=newcmap.N, clip=True)
    
            ax.coastlines()
            gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size':10}        
            d = ax.pcolormesh(ctrl.nav_lon_grid_T, ctrl.nav_lat_grid_T, map, transform = ccrs.PlateCarree(),  cmap = newcmap, norm=divnorm)
            #c = ax.contour(lon, lat, p_value<0.01, transform = ccrs.PlateCarree(), color='k')
            
            ax.text(-300, -60, title[i], rotation='vertical', fontsize=14)

            
        dx = fig.add_subplot(gs[4, 1:3])
        cbar3 = mpl.colorbar.Colorbar(mappable=d, ax=dx, orientation='horizontal',  ticklocation='bottom', cmap=newcmap)
        cbar3.set_label(r'Potential temperature difference [K]', fontsize=12)

        plt.savefig('/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/'+exp+'/'+exp+'_Thetao_patterns.png', bbox_inches='tight')


# ================================================
# Sezione qt_oce (heat flux into ocean)
# ================================================

def global_heatc(heatc, area, mask):
    """
    Derivata temporale del contenuto di calore integrato [J/m2] → flusso [W/m2]

    Parameters
    ----------
    heatc : xarray.DataArray [time, y, x]
        Heat content integrato verticalmente (J/m2).
    area : np.ndarray [y, x]
        Area di ciascuna cella (m2).
    mask : np.ndarray [y, x]
        Maschera booleana (True = oceano).

    Returns
    -------
    years_mid : np.ndarray [time-1]
        Anni centrati tra le differenze.
    dEdt_per_area : np.ndarray [time-1]
        Derivata dE/dt normalizzata per area [W/m2].
    """
    masked = heatc * mask
    total_J = np.nansum(masked * area, axis=(1, 2))  # [time] J

    years = heatc.year.values
    dt = np.diff(years) * 365 * 24 * 3600  # anni → secondi (approssimazione)

    dEdt = np.diff(total_J) / dt           # [time-1] W
    ocean_area = np.nansum(area * mask)
    dEdt_per_area = dEdt / ocean_area      # [time-1] W/m2

    # anni centrati tra le differenze (utile per plottare)
    years_mid = 0.5 * (years[1:] + years[:-1])

    return years_mid, dEdt_per_area

def global_qt_oce(var, area, mask):
    """
    Flusso superficiale qt_oce [W/m2], area-pesato
    var  : [time, y, x] W/m2
    area : [y, x] m2
    mask : [y, x] boolean
    """
    masked = var * mask
    total_flux = np.nansum(masked * area, axis=(1,2))  # W
    ocean_area = np.nansum(area * mask)
    return total_flux / ocean_area  # W/m2


def global_gh_flux(gh_flux, area, mask, nav_lon, nav_lat):
    """
    Calcola il flusso geotermico medio globale [W/m2].
    gh_flux: [time, lat, lon] in mW/m2 (griglia regolare)
    area   : [y, x] in m2 (griglia NEMO)
    mask   : [y, x] boolean (griglia NEMO)
    nav_lon, nav_lat: coordinate della griglia NEMO
    """
    # interpola gh_flux sulla griglia NEMO
    gh_flux = gh_flux.interp(lon=(nav_lon % 360), lat=nav_lat, method="linear")
    gh_wm2 = gh_flux * 1e-3  # converte mW/m2 → W/m2
    masked = gh_wm2 * mask
    total_flux = np.nansum(masked * area)
    ocean_area = np.nansum(area * mask)

    return total_flux / ocean_area  # W/m2

def global_ocean_imbalance(heatc, qt_oce, gh_flux, area, mask, nav_lon, nav_lat):
    """
    Calcola l'imbalance del budget energetico oceanico [W/m2].
    """
    years_mid, dEdt = global_heatc(heatc, area, mask)  # [time-1]
    qt = global_qt_oce(qt_oce[:-1], area, mask)           # accorcia a [time-1]
    gh = global_gh_flux(gh_flux, area, mask, nav_lon, nav_lat)              # scalare → broadcast
    imbalance = dEdt - (qt + gh)
    return years_mid, imbalance


#def plot_qt_oce(ctrl, test, ocean_area, mask, basins, exp):
def plot_qt_oce(test, ocean_area, mask, basins, exp):
    """
    Plot dei flussi qt_oce [W/m2] per bacino e globale
    """
    window_1 = 60  # ultimi anni
    
    fig, ax = plt.subplots(2, 3, figsize=(18,10))
    ax = ax.flatten()
    
    # Ciclo su tutti i bacini
    for i, basin in enumerate(basins):
        mask_b = mask[basin['label']]
        #ctrl_b = global_qt_oce(ctrl[-window_1:], ocean_area, mask_b)
        test_b = global_qt_oce(test[-window_1:], ocean_area, mask_b)
        #diff = test_b - ctrl_b
        #years = ctrl.year.values[-window_1:]
        years = test.year.values[-window_1:]

        #ax[i].plot(years, ctrl_b, label='ctrl', color='g', linewidth=2)
        ax[i].plot(years, test_b, label=exp, color='r', linewidth=2)
        #ax[i].plot(years, diff, label='diff', color='k', linestyle='--')

        ax[i].set_title(basin['label'], fontsize=14)
        ax[i].set_xlabel('Year', fontsize=12)
        ax[i].set_ylabel('Heat flux [W/m²]', fontsize=12)
        ax[i].legend(fontsize=10)
        ax[i].grid(alpha=0.3)

    # elimina pannello vuoto
    fig.delaxes(ax[-1])
    
    plt.suptitle('Surface downward heat flux into ocean [W/m²]', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/{exp}/{exp}_qt_oce.pdf', bbox_inches='tight')

#def plot_ocean_budget(ctrl, test, ocean_area, mask, exp, gh_flux_ctrl, gh_flux_test, nav_lon, nav_lat):
def plot_ocean_budget(test, ocean_area, mask, exp, gh_flux_ctrl, gh_flux_test, nav_lon, nav_lat):
    """
    Plot globale: dE/dt, qt_oce, gh_flux, imbalance e versione con snow.
    Stampa anche i valori medi numerici.
    """
    window_1 = 60
    mask_global = mask['Global']

    # ================= CTRL =================
    # heatc_ctrl = ctrl.heatc[-window_1:]
    # qt_ctrl_raw = ctrl.qt_oce[-window_1:]
    # snow_ctrl = ctrl.snowpre[-window_1:] * 3.34e5  # W/m2

    # # imbalance standard
    # years_ctrl, imbalance_ctrl = global_ocean_imbalance(
    #     heatc_ctrl, qt_ctrl_raw, gh_flux_ctrl,
    #     ocean_area, mask_global, nav_lon, nav_lat
    # )

    # # dEdt e gh flux
    # years_ctrl_dEdt, dEdt_ctrl = global_heatc(heatc_ctrl, ocean_area, mask_global)
    # gh_ctrl = global_gh_flux(gh_flux_ctrl, ocean_area, mask_global, nav_lon, nav_lat)

    # # qt_oce e qt_oce + snow (slice coerente)
    # qt_ctrl = global_qt_oce(qt_ctrl_raw[:-1], ocean_area, mask_global)
    # snow_ctrl = global_qt_oce(snow_ctrl[:-1], ocean_area, mask_global)

    # ================= TEST =================
    heatc_test = test.heatc[-window_1:]
    qt_test_raw = test.qt_oce[-window_1:]
    snow_test = test.snowpre[-window_1:] * 3.34e5  # W/m2

    years_test, imbalance_test = global_ocean_imbalance(
        heatc_test, qt_test_raw, gh_flux_test,
        ocean_area, mask_global, nav_lon, nav_lat
    )

    years_test_dEdt, dEdt_test = global_heatc(heatc_test, ocean_area, mask_global)
    gh_test = global_gh_flux(gh_flux_test, ocean_area, mask_global, nav_lon, nav_lat)

    qt_test = global_qt_oce(qt_test_raw[:-1], ocean_area, mask_global)
    snow_test = global_qt_oce( snow_test[:-1], ocean_area, mask_global)


    # ================= STAMPA =================
    # print(f"\n=== CTRL ({ctrl.attrs.get('sim_name','CTRL')}) ===")
    # print(f" dE/dt mean   [W/m²]: {np.nanmean(dEdt_ctrl):.4f}")
    # print(f" qt_oce mean  [W/m²]: {np.nanmean(qt_ctrl):.4f}")
    # print(f" snow mean [W/m²]: {np.nanmean(snow_ctrl):.4f}")
    # print(f" gh_flux mean [W/m²]: {gh_ctrl:.4f}")
    # print(f" imbalance mean [W/m²]: {np.nanmean(imbalance_ctrl):.4f}")
    

    print(f"\n=== TEST ({exp}) ===")
    print(f" dE/dt mean   [W/m²]: {np.nanmean(dEdt_test):.4f}")
    print(f" qt_oce mean  [W/m²]: {np.nanmean(qt_test):.4f}")
    print(f" qt+snow mean [W/m²]: {np.nanmean(snow_test):.4f}")
    print(f" gh_flux mean [W/m²]: {gh_test:.4f}")
    print(f" imbalance mean [W/m²]: {np.nanmean(imbalance_test):.4f}")

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(9,6))
    #ax.plot(years_ctrl_dEdt, dEdt_ctrl, label='dE/dt ctrl', color='g')
    ax.plot(years_test_dEdt, dEdt_test, label='dE/dt '+exp, color='r')
    #ax.plot(years_ctrl_dEdt, qt_ctrl, label='qt_oce ctrl', color='g', linestyle='--')
    ax.plot(years_test_dEdt, qt_test, label='qt_oce '+exp, color='r', linestyle='--')
    #ax.axhline(gh_ctrl, color='g', linestyle=':', label='gh_flux ctrl')
    ax.axhline(gh_test, color='r', linestyle=':', label='gh_flux '+exp)
    #ax.plot(years_ctrl, imbalance_ctrl, label='imbalance ctrl', color='k')
    ax.plot(years_test, imbalance_test, label='imbalance '+exp, color='k', linestyle='--')

    ax.set_title('Global', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Heat flux [W/m²]', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle('Global ocean heat budget (dE/dt vs qt_oce + gh_flux)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/{exp}/{exp}_ocean_budget.pdf', bbox_inches='tight')


# def load_density(exp):
#     path = '/ec/res4/hpcperm/itcv/analysis/rho_fields_'+exp+'.nc'  # change with rho_fields
#     data = xr.open_mfdataset(path)
#     data.close()
#     return data

# def plotRho(ctrl, test, levels, exp):

#     fig, ax = plt.subplots(1,2, figsize=(20,10))
#     fig.suptitle('Density [kg/m3]', y=0.95, fontsize=16)

#     ax[0].plot(ctrl, -levels/1000, label='ctrl', color='g')
#     ax[0].plot(test, -levels/1000, label=exp, color='r')
#     ax[0].tick_params(labelsize=12)
#     ax[0].set_ylabel('Depth [km]', fontsize=14)

#     ax[1].plot(test - ctrl, -levels/1000, color='k')
#     ax[1].tick_params(labelsize=12)
#     #ax[1].set_xlim(-0.025,0.025)

#     ax[0].legend(fontsize=12)
#     plt.savefig('/ec/res4/hpcperm/itcv/analysis/'+exp+'/'+exp+'_density.pdf', bbox_inches='tight')

# def plotNsquared(ctrl, test, levels, exp):

#     fig, ax = plt.subplots(1,2, figsize=(20,10))
#     fig.suptitle('Brunt Vaisala Frequency [rad/s]', y=0.95, fontsize=16)

#     ax[0].plot(ctrl, -levels, label='ctrl', color='g')
#     ax[0].plot(test, -levels, label=exp, color='r')
#     ax[0].tick_params(labelsize=12)
#     ax[0].set_ylabel('Depth [km]', fontsize=14)

#     ax[1].plot(test - ctrl, -levels, color='k')
#     ax[1].tick_params(labelsize=12)
#     #ax[1].set_xlim(-0.025,0.025)

#     ax[0].legend(fontsize=12)
#     plt.savefig('/ec/res4/hpcperm/itcv/analysis/'+exp+'/'+exp+'_Nsquared.pdf', bbox_inches='tight')

def test_exp(exp):
    #data_ctrl = load_data('tu7b')
    data_test = load_data(exp)

    window_1 = 60  # ultimi 60 anni
    # seleziono gli ultimi 60 anni di entrambe le simulazioni
    #data_ctrl = data_ctrl.sel(year=slice(data_ctrl.year.values[-window_1], data_ctrl.year.values[-1]))
    data_test = data_test.sel(year=slice(data_test.year.values[-window_1], data_test.year.values[-1]))

    #ocean areas
    path = '/ec/res4/scratch/itas/ece4/pi13/'
    areas = xr.open_mfdataset(path+'areas.nc')
    ocean_area = areas['ORCA2-T.srf'].values
    ocean_area[np.isnan(data_test.thetao[0,0])] = np.nan  # controllare se ha senso toglierlo oppure no

    v_mask = data_test.thetao[0]/data_test.thetao[0]
    v_area = (v_mask*ocean_area).sum(axis=(1,2))

    levels = data_test.deptht

    #masks
    Atlantic = dict({'color':'blue', 'label':'Atlantic', 'index':1}) #'lat': [-30,48], 'lon': [280,380]
    Southern = dict({'color':'#920263', 'label':'Southern', 'index':4})
    Arctic = dict({'color':'red', 'label':'Arctic', 'index':5})
    globe = dict({ 'color':'black', 'label':'Global', 'index':0}) # 'lat': [-90,90], 'lon': [0,360],
    IndoPacific = dict({ 'color':'orange', 'label':'IndoPacific', 'index':6}) # 'lat': [-90,90], 'lon': [0,360],
    basins = [globe, Atlantic, IndoPacific, Southern, Arctic]

    path = '/ec/res4/scratch/ccff/ece4/tu7b/'
    ncfile = xr.open_mfdataset(path+'subbasins.nc')

    SouthernMask = np.array(np.zeros([148,180]), dtype=bool)
    SouthernMask[data_test.nav_lat_grid_T<-30] = True
    SouthernMask[np.isnan(data_test.thetao[0,0])] = False # np.nan

    AtlanticMask = np.array(np.zeros([148,180]), dtype=bool) # np.array(ncfile.variables['tmaskatl'])
    AtlanticMask[ncfile.atlmsk.values==True] = True
    AtlanticMask[data_test.nav_lat_grid_T<-30] = False
    AtlanticMask[data_test.nav_lat_grid_T>60] = False
    AtlanticMask[np.isnan(data_test.thetao[0,0])] = False #np.nan # true and nan give same result

    IndoPacificMask = np.array(np.zeros([148,180]), dtype=bool) 
    IndoPacificMask[ncfile.pacmsk.values==True] = True
    IndoPacificMask[ncfile.indmsk.values==True] = True
    IndoPacificMask[data_test.nav_lat_grid_T<-30] = False
    IndoPacificMask[data_test.nav_lat_grid_T>60] = False
    IndoPacificMask[np.isnan(data_test.thetao[0,0])] = False #np.nan

    ArcticMask = np.array(np.zeros([148,180]), dtype=bool) 
    ArcticMask[data_test.nav_lat_grid_T>60] = True
    ArcticMask[np.isnan(data_test.thetao[0,0])] = False #np.nan

    GlobalMask = np.array(np.ones([148,180]), dtype=bool)
    GlobalMask[np.isnan(data_test.thetao[0,0])] = False #np.nan

    mask = dict({'Atlantic': AtlanticMask,'Global': GlobalMask,
                'Southern' : SouthernMask, 'Arctic':ArcticMask, 'IndoPacific':IndoPacificMask})

    # Thetao plot divided by basin
    #thetao_ctrl = global_mean(data_ctrl.thetao.mean(axis=0), ocean_area, v_area)
    thetao_test = global_mean(data_test.thetao.mean(axis=0), ocean_area, v_area)

    # carico heatc, qt_oce, gh_flux

    #qt_ctrl = data_ctrl.qt_oce
    qt_test = data_test.qt_oce

    #gh_path_ctrl = '/ec/res4/scratch/ccff/ece4/tu7b/Goutorbe_ghflux.nc'
    gh_path_test = f'/ec/res4/scratch/itas/ece4/{exp}/Goutorbe_ghflux.nc'
    #gh_flux_ctrl = xr.open_dataset(gh_path_ctrl).gh_flux
    gh_flux_test = xr.open_dataset(gh_path_test).gh_flux

   
    plot_qt_oce(qt_test, ocean_area, mask, basins, exp)
    nav_lon, nav_lat = data_test.nav_lon_grid_T, data_test.nav_lat_grid_T

    plot_ocean_budget(data_test, ocean_area, mask, exp, gh_flux_test, nav_lon, nav_lat)

    #so_ctrl = global_mean(data_ctrl.so.mean(axis=0), ocean_area, v_area)
    so_test = global_mean(data_test.so.mean(axis=0), ocean_area, v_area)
    
    # density
    # density_ctrl = load_density('ctrl')
    # density_test = load_density(exp)

    
    # rho_ctrl = global_mean(density_ctrl.density.mean(axis=0), ocean_area, v_area)
    # rho_test = global_mean(density_test.density.mean(axis=0), ocean_area, v_area)

    # v_mask = density_ctrl.Nsquared[0]/density_ctrl.Nsquared[0]
    # v_area = (v_mask*ocean_area).sum(axis=(1,2))

    # N2_ctrl = global_mean(density_ctrl.Nsquared.mean(axis=0), ocean_area, v_area)
    # N2_test = global_mean(density_test.Nsquared.mean(axis=0), ocean_area, v_area)
    
    # plotNsquared(N2_ctrl, N2_test, density_ctrl.depth_mid, exp)

    # plot for ocean state
    fig, ax = plt.subplots(2,3, figsize=(15,12))
    #ax[0,0].plot(thetao_ctrl, -levels/1000, label='ctrl', color='g')
    ax[0,0].plot(thetao_test, -levels/1000, label=exp, color='r')
    ax[0,0].set_title('Temperature [C]')
    ax[0,0].legend()
    ax[0,0].set_ylabel('Depth [km]')

    #ax[0,1].plot(so_ctrl, -levels/1000, label='ctrl', color='g')
    ax[0,1].plot(so_test, -levels/1000, label=exp, color='r')
    ax[0,1].set_title('Salinity [g/kg]')
    ax[0,1].set_ylabel('Depth [km]')

    # ax[0,2].plot(rho_ctrl, -levels/1000, label='ctrl', color='g')
    # ax[0,2].plot(rho_test, -levels/1000, label=exp, color='r')
    ax[0,2].text(0.5, 0.5, 'Density skipped', ha='center', va='center', fontsize=12)
    ax[0,2].set_title('Density [kg/m3]')
    ax[0,2].set_ylabel('Depth [km]')


    #ax[1,0].plot(thetao_test-thetao_ctrl, -levels/1000, color='k')
    #ax[1,1].plot(so_test-so_ctrl, -levels/1000, color='k')
    #ax[1,2].text(0.5, 0.5, 'Density skipped', ha='center', va='center', fontsize=12)
    # ax[1,2].plot(rho_test-rho_ctrl, -levels/1000, color='k')

    plt.savefig('/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/'+exp+'/'+exp+'_Oceanstate.pdf', bbox_inches='tight')
    
"""

# %%
def load_streamfunction(exp):
    path = '/ec/res4/scratch/itcv/ece4/'+exp+'/output/nemo/'
    data = xr.open_mfdataset(path+exp+'_oce_1m_diaptr3d_*.nc').msftyz.groupby("time_counter.year").mean()
    data = data[:,:,:,:,0] # necessary to remove last dimension = 1
 
    return data

# %%
msftyz_ctrl = load_streamfunction('ctrl')
msftyz_ot01 = load_streamfunction('ot01')
msftyz_ot10 = load_streamfunction('ot10')

# %%
lev_msf = msftyz_ctrl.depthw
lat_msf = msftyz_ctrl.nav_lat #[:,0].values

# %%
msftyz_ctrl

# %%
fig, ax = plt.subplots(1,3, figsize=(20,6))

plt.suptitle('Global Ocean', fontsize=20)
ref_msf = msftyz_ctrl[:,0].sel(year=slice(2010,2039)).mean(dim='year')

clevels = np.arange(-40,45,5)
c = ax[0].contourf(lat_msf,-lev_msf/1000, ref_msf, levels=clevels, extend='both')
ax[0].set_xlim(-30,90)
ax[0].set_xlabel('Latitude', fontsize=16)
ax[0].set_ylabel('Depth [m]', fontsize=16)
ax[0].tick_params(labelsize=14)
ax[0].set_title('msftyz ctrl', fontsize=18)

cb = plt.colorbar(c, ax=ax[0], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

clevels = np.arange(-3,3.5,0.5)
divnorm = colors.TwoSlopeNorm(vmin=-3, vmax=3, vcenter=0)
d = ax[1].contourf(lat_msf,-lev_msf/1000, msftyz_ot01[:,0].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm=divnorm, levels=clevels, cmap='RdBu_r')
ax[1].set_xlim(-30,90)
ax[1].set_xlabel('Latitude', fontsize=16)
ax[1].tick_params(labelsize=14)
ax[1].set_title('ot01', fontsize=18)

cb = plt.colorbar(d, ax=ax[1], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

e = ax[2].contourf(lat_msf,-lev_msf/1000, msftyz_ot10[:,0].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm =divnorm, levels=clevels, cmap='RdBu_r')
ax[2].set_xlim(-30,90)
ax[2].set_xlabel('Latitude', fontsize=16)
ax[2].tick_params(labelsize=14)
ax[2].set_title('ot10', fontsize=18)

cb = plt.colorbar(e, ax=ax[2], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)


# %%
fig, ax = plt.subplots(1,3, figsize=(20,6))

plt.suptitle('Atlantic Ocean', fontsize=20)
ref_msf = msftyz_ctrl[:,1].sel(year=slice(2010,2039)).mean(dim='year')

c = ax[0].contourf(lat_msf,-lev_msf/1000, ref_msf)
ax[0].set_xlim(-30,90)
ax[0].set_xlabel('Latitude', fontsize=16)
ax[0].set_ylabel('Depth [m]', fontsize=16)
ax[0].tick_params(labelsize=14)
ax[0].set_title('msftyz ctrl', fontsize=18)

cb = plt.colorbar(c, ax=ax[0], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

clevels = np.arange(-1.5,1.7,0.2)
divnorm = colors.TwoSlopeNorm(vmin=-1.5, vmax=1.5, vcenter=0)
d = ax[1].contourf(lat_msf,-lev_msf/1000, msftyz_ot01[:,1].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm=divnorm, levels=clevels, cmap='RdBu_r')
ax[1].set_xlim(-30,90)
ax[1].set_xlabel('Latitude', fontsize=16)
ax[1].tick_params(labelsize=14)
ax[1].set_title('ot01', fontsize=18)

cb = plt.colorbar(d, ax=ax[1], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

e = ax[2].contourf(lat_msf,-lev_msf/1000, msftyz_ot10[:,1].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm =divnorm, levels=clevels, cmap='RdBu_r')
ax[2].set_xlim(-30,90)
ax[2].set_xlabel('Latitude', fontsize=16)
ax[2].tick_params(labelsize=14)
ax[2].set_title('ot10', fontsize=18)

cb = plt.colorbar(e, ax=ax[2], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)


# %%
fig, ax = plt.subplots(1,3, figsize=(20,6))

plt.suptitle('IndoPacific Ocean', fontsize=20)
ref_msf = msftyz_ctrl[:,2].sel(year=slice(2010,2039)).mean(dim='year')

c = ax[0].contourf(lat_msf,-lev_msf/1000, ref_msf)
ax[0].set_xlim(-30,90)
ax[0].set_xlabel('Latitude', fontsize=16)
ax[0].set_ylabel('Depth [m]', fontsize=16)
ax[0].tick_params(labelsize=14)
ax[0].set_title('msftyz ctrl', fontsize=18)

cb = plt.colorbar(c, ax=ax[0], orientation='horizontal')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

clevels = np.arange(-1.5,1.7,0.2)
divnorm = colors.TwoSlopeNorm(vmin=-1.5, vmax=1.5, vcenter=0)
d = ax[1].contourf(lat_msf,-lev_msf/1000, msftyz_ot01[:,2].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm=divnorm, levels=clevels, cmap='RdBu_r', extend='both')
ax[1].set_xlim(-30,90)
ax[1].set_xlabel('Latitude', fontsize=16)
ax[1].tick_params(labelsize=14)
ax[1].set_title('ot01', fontsize=18)

cb = plt.colorbar(d, ax=ax[1], orientation='horizontal', extend='both')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)

e = ax[2].contourf(lat_msf,-lev_msf/1000, msftyz_ot10[:,2].sel(year=slice(2010,2039)).mean(dim='year') - ref_msf, norm =divnorm, levels=clevels, cmap='RdBu_r', extend='both')
ax[2].set_xlim(-30,90)
ax[2].set_xlabel('Latitude', fontsize=16)
ax[2].tick_params(labelsize=14)
ax[2].set_title('ot10', fontsize=18)

cb = plt.colorbar(e, ax=ax[2], orientation='horizontal', extend='both')
cb.set_label('Sv', fontsize=12)
cb.ax.tick_params(labelsize=12)


# %% [markdown]
# # Transport (hfbasin)

# %%
def load_transport(exp):
    path = '/ec/res4/scratch/itcv/ece4/'+exp+'/output/nemo/'
    data = xr.open_mfdataset(path+exp+'_oce_1m_diaptr2d_*.nc').hfbasin.groupby("time_counter.year").mean()
    data = data[:,:,:,0]
    return data

# %%
tran_ctrl = load_transport('ctrl')
tran_ot01 = load_transport('ot01')
tran_ot10 = load_transport('ot10')

# %%
lat_tran = tran_ctrl.nav_lat

# %%
tran_ctrl

# %%
def zonalProfile(var, ax):
    ax.plot(lat_tran, var.sel(basin=1, year=slice(2010,2039)).mean(['year']), label='Global', color='k')
    ax.plot(lat_tran, var.sel(basin=2, year=slice(2010,2039)).mean(['year']), label='Atlantic', color='blue')
    ax.plot(lat_tran, var.sel(basin=3, year=slice(2010,2039)).mean(['year']), label='Indian+Pacific', color='green')
    #ax.set_xlim(-30,90)
    #ax.set_ylim(-2.5e15,2.5e15)
    #ax.set_title(str(var), fontsize=15)
    ax.legend()

fig, axes = plt.subplots(1,3, figsize=(20,5))
zonalProfile(tran_ctrl, axes[0])
zonalProfile(tran_ot01, axes[1])
zonalProfile(tran_ot10, axes[2])

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment analysis")
    parser.add_argument("-i", "--expname", type=str, required=True, help="Source experiment name (e.g., aa00).")
    
    args = parser.parse_args()
    
    # Specify the directory name
    directory_name = args.expname

    # Create the directory
    try:
        os.mkdir('/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/'+directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    test_exp(args.expname)



# def compare_thetao_netsfc(thetao, net_sfc, ocean_area, v_area, exp):
#     """
#     Comparison between ocean heat content (thetao) and net surface flux (net_sfc).
#     """
#     years = thetao.year.values
#     gthetao = global_mean(thetao, ocean_area, v_area).mean(axis=1)  # media su tutti i livelli
#     gnet = global_mean(net_sfc, ocean_area, v_area)                 # W/m2
    
#     fig, ax1 = plt.subplots(figsize=(10,6))
#     ax1.plot(years, gthetao, label="Global thetao", color="b")
#     ax2 = ax1.twinx()
#     ax2.plot(years, gnet, label="Global net_sfc", color="r")
    
#     ax1.set_ylabel("θo [K]")
#     ax2.set_ylabel("Net surface flux [W/m²]")
#     ax1.set_xlabel("Year")
#     fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
#     plt.title("Comparison: Global θo vs Net Surface Flux")
#     plt.savefig(f"/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/{exp}/{exp}_thetao_vs_netsfc.pdf", bbox_inches="tight")

# def compare_qtoce_netsfc(qt_oce, net_sfc, ocean_area, mask, years, units="W/m2", exp="exp"):
#     """
#     Compare the qt_oce and net_sfc fluxes (masked over the ocean).
#     """
#     mask_ocean = mask["Global"]
    
#     g_qt = global_qt_oce(qt_oce, ocean_area, mask_ocean, units=units)
#     g_sfc = global_qt_oce(net_sfc, ocean_area, mask_ocean, units=units)
    
#     plt.figure(figsize=(10,6))
#     plt.plot(years, g_qt, label="qt_oce", color="b", linewidth=2)
#     plt.plot(years, g_sfc, label="net_sfc", color="r", linewidth=2)
#     plt.plot(years, g_qt - g_sfc, label="Δ (qt_oce - net_sfc)", color="k", linestyle="--", linewidth=2)
#     plt.xlabel("Year", fontsize=12)
#     plt.ylabel(f"Heat flux [{units}]", fontsize=12)
#     plt.title("Global comparison qt_oce vs net_sfc")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/analysis/{exp}/{exp}_qtoce_vs_netsfc.pdf", bbox_inches="tight")

# da mettere in test_exp
# === NEW ANALYSES ===
    
    # === Calcola net_sfc da OIFS ===
    # oifs_path = f"/ec/res4/scratch/ccff/ece4/{exp}/output/oifs"
    # net_sfc = compute_net_sfc(oifs_path, exp)

    # # Confronto thetao vs net_sfc
    # compare_thetao_netsfc(data_test.thetao, data_test.net_sfc, ocean_area, v_area, exp)

    # Confronto qt_oce vs net_sfc
    # compare_qtoce_netsfc(data_test.qt_oce, data_test.net_sfc, ocean_area, mask, years, exp)
