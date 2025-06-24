#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import make_plumeddata as mp
import matplotlib.cm as cm
# import re_grplot as rgp


TIME = {"s": 1e-15, "ms": 1e-12, "mcs": 1e-9, "ns": 1e-6, "ps": 1e-3, "fs": 1}
TIME_LABEL = {"s": "$s$", "ms": "$m$$s$", "mcs": "μs",
              "ns": "ns", "ps": "ps", "fs": "fs"}

# LaTeX設定を変更
plt.rcParams['text.usetex'] = True  # LaTeXを使用する
plt.rcParams['font.family'] = 'sans-serif'
# Matplotlibのデフォルトフォント
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def MakeFig(rows, cols, grid=True):
    """図の体裁
    """
    plt.style.use('classic')
    plt_dic = {}
    plt_dic['legend.fancybox'] = True
    plt_dic['legend.labelspacing'] = 0.3
    plt_dic['legend.numpoints'] = 1
    plt_dic['figure.figsize'] = [cols*7, rows*3]
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


def get_colvar(infile) -> dict:
    label, _ = os.path.splitext(os.path.basename(infile))
    with open(infile) as o:
        body = o.read()
    # keys = re.match('#! .+\n()', body).group(0).split()[2:]
    # print(f'key: {" ".join(keys)}')

    body = re.findall('#!( .+?\n[\d\s\.\-]+)', body, re.DOTALL)
    data = {}
    time_init = 0.0
    for block in body:
        lines = block.split('\n')
        keys = lines[0].split()[1:]
        print(f'key: {" ".join(keys)}')
        for line in tqdm(lines[1:]):
            # print(line)
            for idx, value in enumerate(line.split()):
                key = keys[idx]
                value = float(value)
                if key not in data:
                    data[key] = [value]
                    continue
                if key == "time":
                    if value == 0.0:
                        time_init = data['time'][-1]
                        print(f'read step: {time_init}')
                    value += time_init
                data[key].append(value)
    return data

def rev_time_unit(times, dt, unit):
    """time: [fs]
    """
    new_times = []
    for time in times:
        new_time = time * dt * TIME[unit]
        new_times.append(new_time)
    return new_times

def plot_xrd(data_dict, y_list, x_label, c_map="summer", save_fig=True,
             base="CV", pdf_true=True):
    name_list = list(data_dict.keys())
    fig_num = len(y_list)
    col_num = int(np.ceil(np.sqrt(fig_num)))
    row_num = int(np.ceil(fig_num / col_num))
    if pdf_true == True:
        # row_num_ = col_num
        # col_num = row_num
        # row_num = row_num_
        outfile = f"{base}.pdf"
    else:
        outfile = f"{base}.png"

    fig = MakeFig(col_num, row_num)
    ax = []

    for color_idx, name in enumerate(name_list):
        X = data_dict[name]['time']
        for fig_idx, title in enumerate(y_list):
            if color_idx == 0:
                ax.append(fig.add_subplot(col_num, row_num, fig_idx+1))
                ax[fig_idx].set_xlabel(x_label)
                y_label = f"CV{fig_idx+1}"
                ax[fig_idx].set_ylabel(y_label)
                # ax[fig_idx].set_title(title)
                func = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
                # ax[fig_idx].yaxis.set_major_formatter(func)
            try:
                y = data_dict[name][title]
                print(f"{title}: {y.mean()}")
            except KeyError:
                continue
            if c_map == 'jet':
                color = cm.jet(1-(color_idx/len(name_list)))
            elif c_map == 'summer':
                color = cm.summer(color_idx/len(name_list))
            elif c_map == 'cool':
                color = cm.cool(color_idx/len(name_list))
            elif c_map == 'Wistia':
                color = cm.Wistia(1-(color_idx/len(name_list)))
            try:
                ax[fig_idx].plot(X, y, label=name, c=color)
            except ValueError:
                X = X[:len(y)]
                ax[fig_idx].plot(X, y, label=name, c=color)
            # ax[fig_idx].legend(loc="best")

    plt.show()
    if save_fig == True:
        fig.patch.set_alpha(0)  # transparent=True -> all
        fig.savefig(outfile)
        print(f"{outfile} created")


if __name__ == "__main__":
    import argparse
    
    par = argparse.ArgumentParser(description="cv plot")
    par.add_argument('-i', '--infiles', help='colvar file', nargs="+",
                     default=['COLVAR'])
    par.add_argument('-k', '--keys', nargs="*",
                     default=['RD22', 'RD33'])
    par.add_argument('-a', '--key_all', action='store_true')
    par.add_argument('-dt', '--timestep', type=float, default=1,
                     help="default=1 fs")
    par.add_argument('-t', '--time_unit', choices=list(TIME.keys()),
                     default='mcs')

    par.add_argument('-c', '--c_map', default='Wistia',
                     choices=['jet', 'summer', 'cool', "Wistia"])
    par.add_argument('-v', '--verbose', action='store_true', default=True)
    par.add_argument('-s', '--steps', type=int, nargs=2, default=[0, -1],
                     help='step length')
    par.add_argument('-o', '--outfile', help='output name')
    args = par.parse_args()

    x_label = f'time/{TIME_LABEL[args.time_unit]}'
    # y_labels = {k: f"CV{i}" for i, k in enumerate(args.keys())}
    
    
    
    data = {}
    for infile in tqdm(args.infiles):
        label, ext = os.path.splitext(os.path.basename(infile))
        data[label] = get_colvar(infile)
        time = np.array(data[label]['time'], dtype=int)
        if args.steps[1] == -1:
            t_idx = np.where(args.steps[0] <= time)[0]
            base = f"CV_{args.steps[0]}-{time.max()}"
        else:
            t_idx = np.where((args.steps[0] <= time) &
                             (args.steps[1] >= time))[0]
            base = f"CV_{args.steps[0]}-{args.steps[1]}"
        # print({key: np.array(value)[t_idx]
        #        for key, value in data[label].items()})
        for key, value in data[label].items():
            try:
                data[label][key] = np.array(value)[t_idx]
            except IndexError:
                new_value = np.empty(time.size)
                new_value[-len(value):] = value
                data[label][key] = np.array(new_value)[t_idx]
        # data[label] = {key: 
        #                for key, value in data[label].items()}
        data[label]['time'] = rev_time_unit(data[label]['time'],
                                            args.timestep, args.time_unit)

    if args.key_all == True:
        args.keys = list(data[label].keys())
        args.keys.remove('time')
    plot_xrd(data, args.keys, x_label, c_map=args.c_map, base=base)
