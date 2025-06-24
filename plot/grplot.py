#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def get_csv(infile, y_std=21):
    data = pd.read_csv(infile)
    r = np.array(data.iloc[:, 0], dtype=float)
    gr = np.array(data.iloc[:, 1], dtype=float) - y_std
    gr = gr[np.argsort(r)]
    r = r[np.argsort(r)]
    return r, gr

def get_gr(infile):
    with open(infile) as o:
        body = o.read().split()[29:]
    data = np.array(body).astype(float).reshape(-1, 3)
    r = data[:, 0].ravel()
    gr = data[:, 1].ravel()
    cn = data[:, 2].ravel()
    return r, gr, cn

        
def MakeFig(rows, cols, grid=True):
    """図の体裁
    """
    plt.style.use('classic')
    plt_dic = {}
    plt_dic['legend.fancybox'] = True
    plt_dic['legend.labelspacing'] = 0.3
    plt_dic['legend.numpoints'] = 1
    plt_dic['figure.figsize'] = [cols*10+2, rows*3]
    plt_dic['font.size'] = 12
    plt_dic['legend.fontsize'] = 14
    plt_dic['axes.labelsize'] = 16
    plt_dic['xtick.major.size'] = 5
    plt_dic['xtick.minor.size'] = 2
    plt_dic['ytick.major.size'] = 5
    plt_dic['ytick.minor.size'] = 2
    plt_dic['xtick.direction'] = 'in'
    plt_dic['savefig.bbox'] = 'tight'
    plt_dic['savefig.dpi'] = 150
    plt_dic['savefig.transparent'] = False
    plt_dic['axes.grid'] = grid
    plt_dic["figure.subplot.left"] = 0.1
    plt_dic["figure.subplot.right"] = 0.90
    plt_dic["figure.subplot.bottom"] = 0.1
    plt_dic["figure.subplot.top"] = 0.9
    plt_dic['figure.subplot.hspace'] = 0.4
    wspace = 0.3
    if wspace > 1:
        plt_dic['figure.subplot.wspace'] = 1
    else:
        plt_dic['figure.subplot.wspace'] = wspace
    plt.rcParams.update(plt_dic)
    fig = plt.figure()
    if grid is True:
        plt_dic['axes.grid'] = True
    return fig

def plot_gr(gr_dict, x_label, y_label, c_map, save_fig=None, legend_off=False):
    fig = MakeFig(1, 1, True)
    ax = fig.add_subplot(1, 1, 1)

    color_idx = 0
    for name, gr_data in gr_dict.items():
        X = gr_data['X']
        y = gr_data['y']
        if c_map == 'jet':
            color = cm.jet(1-(color_idx/len(gr_dict)))
        elif c_map == 'summer':
            color = cm.summer(color_idx/len(gr_dict)*0.9)
        elif c_map == 'cool':
            color = cm.cool(color_idx/len(gr_dict))
        ax.plot(X, y, c=color, label=name)
        # ax.vlines(2.75, 0, 10, colors="k")
        color_idx += 1
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    if legend_off != True:
        ax.legend(bbox_to_anchor=(1.3, 1.1), frameon=False)
    # ax.legend(loc=(0.6, 0), frameon=False)
    plt.show()

    if save_fig != None:
        fig.patch.set_alpha(0)  # transparent=True -> all
        fig.savefig(save_fig)
        print(f"{save_fig} created")

def read_debag(infile):
    return np.mean(np.load(infile), axis=0)


if __name__ == "__main__":
    import argparse
    from glob import glob
    import os

    par = argparse.ArgumentParser(description="test")
    par.add_argument('-i', '--infiles', default=None, nargs='*',
                     help="gr files")
    par.add_argument('-d', '--debag', action='store_true', help="debag")
    par.add_argument('-sf', '--save_fig', default='Gr.pdf',
                     help='fig file name')
    par.add_argument('-l', '--legend', action="store_true",
                     help='legend on/off')
    par.add_argument('-c', '--c_map', choices=['jet', 'summer', 'cool'],
                     default='summer')
    par.add_argument('-r', '--r_width', nargs='+', type=float, default=[0])
    par.add_argument('-y', '--y_height', type=float, default=0.25)
    args = par.parse_args()

    if args.infiles == None:
        args.infiles = glob('./*.gr')
        args.infiles.sort()
    
        
    gr_data = {}
    cnt = 0
    for infile in tqdm(args.infiles):
        name, ext = os.path.splitext(os.path.basename(infile))
        if ext == '.gr':
            r, gr, cn = get_gr(infile)
        elif ext == '.csv':
            r, gr = get_csv(infile)

        gr = gr[r >= args.r_width[0]]
        r = r[r >= args.r_width[0]]

        if len(args.r_width) == 2:
            gr = gr[r <= args.r_width[1]]
            r = r[r <= args.r_width[1]]

        gr_data[name] = {}
        gr_data[name]['X'] = r
        gr_data[name]['y'] = gr + cnt
        cnt += args.y_height

    plot_gr(gr_data, r'$r$/$\AA$', r'$G$($r$)', args.c_map, args.save_fig,
            legend_off=args.legend)
