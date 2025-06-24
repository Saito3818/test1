#!/usr/bin/env python

import re
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.cm as cm
import os
# import cv_time_plot as ctp
import numpy as np


UNIT = {}
UNIT["real"] = {"Step": "",
                "Time": "",
                "Dt": "",
                "Temp": "K",
                "TotEng": "kcal/mol",
                "PotEng": "kcal/mol",
                "KinEng": "kcal/mol",
                "Press": "atm",
                "Volume": "${\\AA^3}$",
                "Cella": "${\\rm \\AA}$",
                "Cellb": "${\\rm \\AA}$",
                "Cellc": "${\\rm \\AA}$",
                "CellAlpha": "deg.",
                "CellBeta": "deg.",
                "CellGamma": "deg.",
                "Density": "${g}$/${cm^3}$",
                "Enthalpy": '',
                "E_pair": '',
                "E_mol": '',
                "c_2": '',
                "c_4": ''}
UNIT["metal"] = {"Step": "",
                 "Temp": "K",
                 "TotEng": "eV",
                 "PotEng": "eV",
                 "Press": "bars",
                 "Volume": "${\\AA^3}$",
                 "Density": "${g}$/${cm^3}$",
                 "Cella": "Ang",
                 "Cellb": "Ang",
                 "Cellc": "Ang",
                 "CellAlpha": "deg",
                 "CellBeta": "deg",
                 "CellGamma": "deg"}
TIME = {"s": 1e-15, "ms": 1e-12, "ns": 1e-6, "ps": 1e-3, "fs": 1}

def GetLog(logfile):
    """read logfile
    """
    # name, ext = os.path.splitext(logfile)
    # if ext != '.log':
    #     print(f'{ext.split(".")[1]} is unabel to read.')
    #     exit()
    with open(logfile) as o:
        body = o.read()
        try:
            units = UNIT[re.findall("units +(\w+?)\n", body)[0]]
        except IndexError:  # .out
            units = UNIT['real']
        heads = [h.split() for h in re.findall('Step.+?\n', body)]
        body = re.findall('Step.+?\n([\s\d\-\.e\+]+)', body, re.DOTALL)
    if len(body) == 0:
        print(f"{logfile} is broken")
    data = {}
    keys = {}
    header = np.unique([h for h_ in heads for h in h_])
    for key in header:
        keys[key] = f"{key} ({units[key]})"
        data[key] = []
    l = 0
    for i, b in enumerate(body):
        for j, key in enumerate(heads[i]):
            data[key] += [float(d) for d in b.split()[j::len(heads[i])]]
        for key in [h for h in header if h not in heads[i]]:
            data[key] += [None] * int(len(b.split()) / (j + 1))
    print(f'{logfile} is read.')
    print(f"key = [{' '.join(header)}]")
    return keys, data

def rev_time_unit(times: list, dt: float, unit: str):
    """time: time list [fs], dt: timestep [fs],
       unit: choices={s, ms, ns, ps, fs}
    """
    new_times = []
    for time in times:
        new_time = time * dt * TIME[unit]
        new_times.append(new_time)
    return new_times

def MakeFig(rows, cols, keys, grid=True):
    """図の体裁
    """
    import numpy as np
    plt.style.use('classic')
    plt_dic = {}
    plt_dic['legend.fancybox'] = True
    plt_dic['legend.labelspacing'] = 0.3
    plt_dic['legend.numpoints'] = 1
    plt_dic['figure.figsize'] = [cols*5+2*keys, rows*5]
    plt_dic['font.size'] = 12
    plt_dic['legend.fontsize'] = 12
    plt_dic['axes.labelsize'] = 16
    plt_dic['xtick.major.size'] = 5
    plt_dic['xtick.minor.size'] = 2
    plt_dic['ytick.major.size'] = 5
    plt_dic['ytick.minor.size'] = 2
    plt_dic['xtick.major.width'] = 1.0
    plt_dic['ytick.major.width'] = 1.0
    plt_dic['xtick.direction'] = 'in'
    plt_dic['savefig.bbox'] = 'tight'
    plt_dic['savefig.dpi'] = 150
    plt_dic['savefig.transparent'] = False
    plt_dic['axes.grid'] = grid
    plt_dic["figure.subplot.left"] = 0.12
    plt_dic["figure.subplot.right"] = 0.9
    plt_dic["figure.subplot.bottom"] = 0.1
    plt_dic["figure.subplot.top"] = 0.9
    plt_dic['figure.subplot.hspace'] = 0.4
    plt_dic['figure.subplot.wspace'] = 0.4 + 0.2 * (keys - 1)
    if plt_dic['figure.subplot.wspace'] >= 1:
        plt_dic['figure.subplot.wspace'] = 0.999
    plt.rcParams.update(plt_dic)
    fig = plt.figure()
    if grid is True:
        plt_dic['axes.grid'] = True
    return fig

def make_patch_spines_invisible(ax):
    """フレームの設定
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
def FigData(ydata, host):
    """軸の生成
    """
    ax = [host]
    for i in range(len(ydata) - 1):
        ax.append(host.twinx())
    if len(ydata) > 2:
        for i in range(2, len(ydata)):
            ax[i].spines["right"].set_position(("axes", 1 + (i-1)*0.20))
            make_patch_spines_invisible(ax[i])
            ax[i].spines["right"].set_visible(True)
        rmargin = 1.0 - (0.11 * (len(ydata) - 2))
    else:
        rmargin = 0.9
    if len(ydata) == 1:
        ax[0].grid()
    return rmargin, ax

def PlotData(data, key, ax, y_labels, steps, timestep, time_unit,
             p='-', c_map='summer'):
    """プロット
    """
    # clst = ["r", "orange", "y", "g", "mediumturquoise", "b", "violet",
    #         "purple", "m", "pink"]

    x = np.array(data['Step'], dtype=int)
    idx = np.where(x >= steps[0])[0]
    if len(steps) == 2:
        idx = np.where((x >= steps[0]) & (x <= steps[1]))[0]
    if len(idx) == 0:
        print(f"ERROR: args.steps = {steps} wrong!")
        exit()

    x = np.array(rev_time_unit(x, timestep, time_unit))
    x_label = f'time({time_unit})'
    ax[0].set_xlabel(x_label)
    for i, k in enumerate(key):
        try:
            y = np.array(data[k])[idx]
        except IndexError:

            idx = idx[:len(data[k])]
            x = x[:]
            y = np.array(data[k])[idx]

            
        # ax[i].yaxis.set_major_formatter(tick.FormatStrFormatter('%i'))
        if c_map == 'jet':
            color = cm.jet(1-(i/len(key)))
        elif c_map == 'summer':
            color = cm.summer(i/len(key))
        elif c_map == 'cool':
            color = cm.cool(i/len(key))
        fix, = ax[i].plot(x[idx], y, p, color=color)
        ax[i].set_ylabel(y_labels[k])
        print(f"{k}={np.mean(y):.4f}")
        if len(key) > 1:       # y軸が複数のとき
            ax[i].yaxis.label.set_color(fix.get_color())
            tkw = dict(size=4, width=1.5)
            ax[i].tick_params(axis='y', colors=fix.get_color(), **tkw)

if __name__ == "__main__":
    import argparse
    par = argparse.ArgumentParser(description="plot logfile\
    careful!: Not argument of logfiles")
    par.add_argument('-i', '--infiles', default=[],
                     nargs="+", help="default: currend directly")
    par.add_argument('-k', '--key', nargs='+',
                     default=['Temp', 'Volume'])
    par.add_argument('-dt', '--timestep', type=float, default=1,
                     help="default=1 fs")
    par.add_argument('-t', '--time_unit', choices=TIME.keys(),
                     default='ns')
    par.add_argument('-s', '--steps', nargs='+', default=[0], type=int,
                     help="[init last] or [init], default=[0]")
    par.add_argument('-c', '--c_map', choices=['jet', 'summer', 'cool'],
                     default='jet')
    par.add_argument("-o", "--outfile", action="store_true",
                     help="outfile->log.png")
    args = par.parse_args()

    data = {}
    keys = {}
    nlst = []

    if len(args.infiles) == 0:
        args.infiles = [f for f in os.listdir("./")
                        if os.path.splitext(f)[1] == ".log"]
    for f in args.infiles:
        name, ext = os.path.splitext(f)
        keys[name], data[name] = GetLog(f)

        nlst.append(name)

    cols = int(np.ceil(np.sqrt(len(data))))
    lows = int(np.ceil(len(data)/cols))
    fig = MakeFig(cols, lows, len(args.key))
    nlst.sort()
    for i, name in enumerate(nlst):
        host = fig.add_subplot(cols, lows, i+1)
        rmargin, ax = FigData(args.key, host)
        fig.subplots_adjust(right=rmargin)
        PlotData(data[name], args.key, ax, keys[name],
                 args.steps, args.timestep, args.time_unit, c_map=args.c_map)
        # ax[0].set_xticklabels(np.linspace(0, 8000000, 5).astype(int))
        plt.locator_params(axis='x', nbins=6)
        plt.title(name, y=1)
    if args.outfile == False:
        fig.patch.set_alpha(0)  # transparent=True -> all
        plt.savefig("log.png")
    plt.show()

        
