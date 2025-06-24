#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import cm
import hills_plot as hp
# import re_grplot as rgp
import cv_hist_plot as chp
import os
import warnings
import re

# POINTS = {}
# POINTS['91ns'] = [40.659222, 65.646520]
# POINTS['620ns'] = [34.095181, 53.211044]
# POINTS['879ns'] = [63.350304, 96.752050]
# POINTS['882ns'] = [34.605985, 88.348899]
# POINTS['2009ns'] = [38.968805, 16.508786]
# POINTS['2111ns'] = [76.768636, 107.504512]
# POINTS['2147ns'] = [56.297173, 101.610739]
# POINTS['2217ns'] = [60.096956, 107.352284]
# POINTS['2524ns'] = [32.285409, 28.449242]

def set_mesh(infile, xlabel, ylabel, zlabel):
    key_list, data = hp.get_fesdat(infile)
    if xlabel == None:
        xlabel = key_list[0]
    if ylabel == None:
        ylabel = key_list[1]
    if zlabel == None:
        zlabel = key_list[2]

    X = data[xlabel]
    Y = data[ylabel]
    Z = data[zlabel]
    X_uniq, X_index = np.unique(data[xlabel],
                                return_index=True)
    Y_uniq, Y_index = np.unique(data[ylabel],
                                return_index=True)
    if X_index[1] == 1:
        X = X.reshape(Y_uniq.size, X_uniq.size)
        Y = Y.reshape(Y_uniq.size, X_uniq.size)
        Z = Z.reshape(Y_uniq.size, X_uniq.size)
    elif Y_index[1] == 1:
        X = X.reshape(X_uniq.size, Y_uniq.size)
        Y = Y.reshape(X_uniq.size, Y_uniq.size)
        Z = Z.reshape(X_uniq.size, Y_uniq.size)
    return xlabel, ylabel, zlabel, X, Y, Z

def plot_hills(data_dict, x_label, y_label, z_label, c_map, z_width,
               save_fig=None, contourf=False, levels=50):
    name_list = list(data_dict.keys())
    fig_list = list(data_dict[name_list[0]].keys())
    z_ticks = np.linspace(z_width[0], z_width[1], levels)
    fig_idx = 1

    warnings.simplefilter('ignore', RuntimeWarning)
    for name in tqdm(data_dict.keys()):
        gr_dict = data_dict[name]
        fig = rgp.MakeFig(1, len(fig_list))
        for fig_idx, fig_name in enumerate(fig_list):
            ax = fig.add_subplot(1, len(fig_list), fig_idx+1)
            X = gr_dict[fig_name]['X']
            Y = gr_dict[fig_name]['Y']
            Z = gr_dict[fig_name]['Z']

            if contourf == True:
                img = ax.contourf(X, Y, Z, cmap=c_map, levels=z_ticks)
            else:
                img = ax.contour(X, Y, Z, cmap=c_map, levels=z_ticks)
            # cbar.set_ticks(z_ticks)
            x_dim = X.max() - X.min()
            y_dim = Y.max() - Y.min()
            dim_max = max(x_dim, y_dim)
            ax.set_xlim(X.min(), X.min()+dim_max)
            ax.set_ylim(Y.min(), Y.min()+dim_max)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f'{fig_name}: {name}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            # plt.show()
        plt.colorbar(img)  # カラーバーを表示
        # fig_idx += 1
        # ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        if save_fig != None:
            os.makedirs("png", exist_ok=True)
            fig.patch.set_alpha(0)  # transparent=True -> all
            fig.savefig(f'png/{name}.png')
            # print(f"\rpng/{name}.png created", end='\n')

if __name__ == '__main__':
    import argparse
    par = argparse.ArgumentParser(description="test")
    par.add_argument('-i', '--infiles', nargs='*', default=None,
                     help='fes.dat')
    par.add_argument('-t', '--timestep', type=float, default=1,
                     help='default: 1 ns')
    par.add_argument('-x', '--xlabel', default=None)
    par.add_argument('-y', '--ylabel', default=None)
    par.add_argument('-z', '--zlabel', default=None)
    par.add_argument('-sf', '--save_fig', default='Gr.png',
                     help='fig file name')
    par.add_argument('-c', '--c_map', choices=['jet', 'summer', 'cool'],
                     default='jet')
    par.add_argument('-e', '--energy', type=float, default=None,
                     help="max energy")
    par.add_argument('-o', '--outfile', help='output name',
                     default='hills_contour.pdf')
    par.add_argument('-a', '--ave_true', action='store_true',
                     help='Which do culucurate cumulative average')
    # par.add_argument('-p', '--point', action='store_true',
    #                  help="write point")
    par.add_argument('-cf', '--contourf', default=False, action="store_true")
    args = par.parse_args()

    if args.infiles == None:
        infiles = {int(os.path.splitext(infile)[0].split('_')[1]): infile
                   for infile in glob('fes_*.dat')}        
        args.infiles = [infiles[key] for key in sorted(infiles)]
    
    if len(args.infiles) == 1:
        xlabel, ylabel, zlabel, X, Y, Z = set_mesh(args.infiles[0], args.xlabel,
                                                   args.ylabel, args.zlabel)
        if args.energy != None:
            Z[Z > args.energy] = args.energy
        chp.plot_2dim(X, Y, Z, "CV1", "CV2",
                      title="",
                      c_map=args.c_map, save_fig=args.outfile,
                      contourf=args.contourf)

    else:  # infileが複数
        data_dict = {}
        if args.ave_true == True:
            fig_list = ['Cumulative FES', 'Cumulative average FES']
        else:
            fig_list = ['Cumulative FES']
        fig_idx = 0
        z_width = [0, 0]
        print("Reading files...")
        for infile in tqdm(args.infiles):
            step = int(os.path.splitext(os.path.basename(infile))[0]
                       .split('_')[1])
            name = "{:.1f} ns".format(step * args.timestep)
            xlabel, ylabel, zlabel, X, Y, Z = set_mesh(infile, args.xlabel,
                                                       args.ylabel, args.zlabel)
            # key_list, data = hp.get_fesdat(infile)
            data_dict[name] = {}

            if Z.min() < z_width[0]:
                z_width[0] = Z.min()
            if Z.max() > z_width[1]:
                z_width[1] = Z.max()

            for fig_name in fig_list:
                data_dict[name][fig_name] = {}
                data_dict[name][fig_name]['X'] = X
                data_dict[name][fig_name]['Y'] = Y
                if fig_name == 'Cumulative average FES':
                    fig_idx += 1
                    if fig_idx == 1:
                        cum_sum = Z
                    else:
                        cum_sum = cum_sum + Z
                    data_dict[name][fig_name]['Z'] = cum_sum / fig_idx
                else:
                    data_dict[name][fig_name]['Z'] = Z
        # print("\rfinished to read!", end='')
        print("Ploting...")
        plot_hills(data_dict, xlabel, ylabel, zlabel,
                   args.c_map, z_width, save_fig=args.save_fig,
                   contourf=args.contourf)
