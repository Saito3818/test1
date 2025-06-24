#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
# import itertools as it

par = argparse.ArgumentParser(description="test")
par.add_argument('lammpstrj', nargs="+")
par.add_argument('-v', '--values', type=str, nargs="+", default=["Ca"],
                 help="atom symbol: Ca, O, H, C")
par.add_argument('-l', '--layer', type=int, default=0, help="layer atom")
par.add_argument('-z', '--z_bins', type=float, default=0.1)
par.add_argument('-s', '--s_bins', type=int, default=10000)
par.add_argument('-i', '--init', type=int, default=400000)
par.add_argument('-f', '--finish', type=int, default=500000)
par.add_argument('-o', '--outpng', action="store_true", help="outfile basename")
par.add_argument('-d', '--dpi', type=int, default=380)
par.add_argument('--axis', choices=["x", "y", "z"], default="z")
args = par.parse_args()

class PLOTZMESH():
    def __init__(self, lammpstrj):
        lows = int(np.ceil(np.sqrt(len(lammpstrj)+len(args.values))))
        cols = int(np.ceil((len(lammpstrj)+len(args.values))/lows))
        fig = self.makefig(lows, cols)
        axis = {"x": 0, "y": 1, "z": 2}[args.axis]
        for i, trj in enumerate(lammpstrj):
            self.getdata(trj, axis)
            for j, v in enumerate(args.values):
                self.name = f"{os.path.splitext(trj)[0]}_{v}"
                ax = fig.add_subplot(lows, cols, i*len(args.values)+j+1)
                self.plotdata(fig, ax, v)
        plt.show()
        if args.outpng == True:
            fig.patch.set_alpha(0)
            outpng = f"zstep_{args.init}-{args.finish}.png"
            fig.savefig(outpng, dpi=args.dpi)
            print(f"{outpng} crated")

    def getdata(self, trj, axis):
        print(f"\r{trj} read start")
        with open(trj) as o:
            body = o.readlines()
        num = int(body[3])
        self.times = {int(t): i for i, t in enumerate(body[1::num+9])
                      if (int(t) % args.s_bins) == 0
                      and int(t) >= args.init and int(t) <= args.finish}
        idxs = list(self.times.values())
        self.atoms = np.array([body[(num+9)*i+9:(num+9)*(i+1)][j].split()
                               for i in idxs for j in range(args.layer, num)],
                              dtype=object)[:, [2, axis+3]]
        self.atoms[:, 1] = self.atoms[:, 1].astype(float)
        print(f"{trj} was read!      ")
        
        
    def makefig(self, lows, cols):
        plt.style.use('classic')
        plt_dic = {}
        plt_dic['legend.fancybox'] = True
        plt_dic['legend.labelspacing'] = 0.3
        plt_dic['legend.numpoints'] = 1
        plt_dic['figure.figsize'] = [lows*10, cols*3]
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
        plt_dic['axes.grid'] = True
        plt_dic["figure.subplot.left"] = 0.12
        plt_dic["figure.subplot.right"] = 0.9
        plt_dic["figure.subplot.bottom"] = 0.1
        plt_dic["figure.subplot.top"] = 0.9
        plt_dic['figure.subplot.hspace'] = 0.4
        plt_dic['figure.subplot.wspace'] = 0.5
        plt.rcParams.update(plt_dic)
        fig = plt.figure()
        return fig

    def plotdata(self, fig, ax, value):
        xlabel = "step"
        ylabel = f"{args.axis} " + "${\\AA}$"
        # color = ["black", "b", "royalblue", "cornflowerblue", "dodgerblue", 
        #          "deepskyblue", "", "darkturquoise", "g", "y",
        #          "orange", "r", "darkred", "purple", "m", "pink"]
        if self.atoms.shape[0] == 0:
            print("step is too short!")
            exit()
        atoms = self.atoms[self.atoms[:, 0] == value][:, 1]
        atoms = atoms.reshape(len(list(self.times.keys())), -1)
        bins = np.arange(self.atoms[:, 1].min(), self.atoms[:, 1].max(),
                         args.z_bins)
        y = bins[:-1] + (bins[1] - bins[0])/2
        X, Y = np.meshgrid(np.array(list(self.times.keys())), y)
        Z = np.empty(X.shape)
        for i, a in enumerate(atoms):
            Z[:, i], _ = np.histogram(a, bins=bins)
        # level = np.arange(Z.max()+1).astype(int)
        # if len(color) <= Z.max():
        #     level = np.linspace(0, Z.max()+1, len(color)).astype(int)
        # ax.contour(X, Y, Z, cmap="jet")
        # ax.clabel(CS, inline=True)
        CSf = ax.contourf(X, Y, Z, cmap="jet")
        cbar = fig.colorbar(CSf, aspect=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar.ax.set_ylabel("frequency")
        plt.title(self.name, y=1)
            
PLOTZMESH(args.lammpstrj)
