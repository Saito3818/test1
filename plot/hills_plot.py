#!/usr/bin/env python
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class FES():
    def __init__(self, fesdats):
        # self.steps = np.empty(len(fesdats), dtype=int)
        self.data = {}
        for fesdat in fesdats:
            step = int(re.search("\d+", fesdat).group(0)) + 1
            self.data[step] = self.read_fesdat(fesdat)
        outfile = "hoge.pdf"
        self.plot_fes(outfile)

    def read_fesdat(self, fesdat):
        with open(fesdat) as o:
            lines = o.readlines()
        self.CV = lines[0].split()[2]
        data = np.array([line.split() for line in lines[5:]], dtype=float)
        data = data.reshape(-1, 3)
        data_dict = {}
        data_dict['X'] = data[:, 0]  # CV
        data_dict['Y'] = data[:, 1]  # file.free
        return data_dict

    def plot_fes(self, outfile):
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        color_idx = 0
        fig.subplots_adjust(right=0.8,)
        for step, data in self.data.items():
            color = cm.jet(color_idx/len(self.data))
            ax.plot(data['X'], data['Y'], label=f"{step} ns", c=color)
            color_idx += 1
        plt.legend(loc=(1.05, -0.05))
        ax.set_xlabel(self.CV, fontsize=13)
        ax.set_ylabel("Energy [kcal/mol]", fontsize=13)
        plt.show()
        fig.savefig(outfile)
        print(f"{outfile} created")


import argparse

par = argparse.ArgumentParser(description="test")
par.add_argument('-n', '--n_bin', type=int)
par.add_argument('-m', '--max_value', type=int)
args = par.parse_args()
fesdats = [f"hills/fes_{i}.dat" for i in range(0, args.max_value, args.n_bin)]
FES(fesdats)
