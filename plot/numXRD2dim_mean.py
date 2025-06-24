#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import make_plumeddata as mp
import matplotlib.cm as cm
import re_grplot as rgp

# http://www.pnas.org/cgi/doi/10.1073/pnas.1803919115
# 全原子でXRD 強度を求めるプログラム。

# mark = ["-r", "-g", "-b", "-c", "-m", "-y"]
# color = ["red", "green", "blue", "cyan", "magenta", "yellow"]
ELEMENT = ["Ca", "C", "O", "O", 'H']



# 原子散乱因子と、散乱定数Q を設定
# lam = 1.5406
# # Ttheta = np.linspace(15, 80, 651)
# Q = 4 * np.pi * np.sin(Ttheta / 360 * np.pi) / lam

# fsi = np.zeros_like(Ttheta)
# fo = np.zeros_like(Ttheta)
# for idx, q in enumerate(Q):  # atom scattering factors
#     Ar = (q / 4 / np.pi)**2  # Ar = (q / 4pi) ** 2
#     fsi[idx] = mp.calc_atom_factor(q, 'C')
#     fo[idx] = mp.calc_atom_factor(q, 'Ca')

def remove_labelname(label):
    label = re.sub("_(\d)", "_{\g<1>}", label)
    label = re.sub("g", "/", label)
    label = re.sub("\((.+)\)", "(${\g<1>}$)", label)
    return label


class XRD():
    # ファイル読み込み、導出したXRDピークをプロットする。
    def __init__(self, Lambda, xrd_ok):
        self.Lambda = Lambda
        self.angle = np.empty(0)
        self.elem_types = np.empty(0)
        self.atoms = np.empty((0, 6), dtype=object)  # id type element x y z
        if xrd_ok == 'COMP':
            self.xrd_list = ['XRD', 'ND']
        else:
            self.xrd_list = [xrd_ok]
        self.coords = {xrd: {} for xrd in self.xrd_list}
        self.factors = {xrd: {} for xrd in self.xrd_list}

            
    def get_trjfile(self, infile, element, step_bin, types=None, Rc=None):
        with open(infile) as f:
            lines = f.readlines()
        num_atoms = int(lines[3])
        num_steps = len(lines) // (num_atoms + 9)
        steps = np.arange(0, num_steps, step_bin) * (num_atoms + 9)

        # ITEM: ATOMS id type element x y z vx vy vz
        indexs = lines[8].split()[2:]
        elem_idx = indexs.index("element")
        try:
            x_idx = indexs.index("x")
            y_idx = indexs.index("y")
            z_idx = indexs.index("z")
        except ValueError:
            x_idx = indexs.index("xu")
            y_idx = indexs.index("yu")
            z_idx = indexs.index("zu")
        xyz_idx = [x_idx, y_idx, z_idx]

        step_idx = 0
        for idx in tqdm(steps):
            # idx = step_idx * (num_atoms + 9)
            box = [line.split() for line in lines[idx+5:idx+8]]
            box = np.array(box, dtype=float)
            a, b, c = (box[:, 1] - box[:, 0]).ravel()
            if box.shape[1] == 2:  # rectangle
                alpha = beta = gamma = 90
            else:
                xy, xz, yz = box[:, 2]
                xlo = box[0, 0] - np.min([0.0, xy, xz, xy+xz])
                xhi = box[0, 1] - np.max([0.0, xy, xz, xy+xz])
                ylo = box[1, 0] - np.min([0.0, yz])
                yhi = box[1, 1] - np.max([0.0, yz])
                lx = xhi - xlo
                ly = yhi - ylo
                lz = box[2, 1] - box[2, 0]
                a = lx
                b = np.sqrt(ly**2 + xy**2)
                c = np.sqrt(lz**2 + xz**2 + yz**2)
                alpha = np.arccos((xy*xz + ly*yz)/(b*c)) * 180/np.pi
                beta = np.arccos(xz/c) * 180/np.pi
                gamma = np.arccos(xy/b) * 180/np.pi
            self.M = mp.CalcMatrix(a, b, c, alpha, beta, gamma)
            self.M_ = np.linalg.inv(self.M)

            atoms = [line.split() for line in
                     lines[idx+9:idx+9+num_atoms]]
            
            self.atoms = np.array(atoms, dtype=object)
            self.atoms[:, xyz_idx] = self.atoms[:, xyz_idx].astype(float)

            if self.elem_types.size == 0:
                if types != None:
                    element = [str(idx) for idx in types]
                    type2elem = {}
                    elem_idx = indexs.index("type")
                self.elem_types = self.atoms[:, elem_idx]
                elem_idxs = np.empty(0, dtype=int)
                for elem in element:
                    elem_id = np.where(self.elem_types == elem)[0]
                    if types != None:
                        type2elem[elem] = self.atoms[elem_id[0],
                                                     indexs.index("element")]
                    elem_idxs = np.append(elem_idxs, elem_id)
                self.elem_types = self.elem_types[elem_idxs]
            # elem_mask = np.isin(self.atoms[:, elem_idx], element)
            xyz = self.atoms[elem_idxs, :][:, xyz_idx] @ self.M_
            distance = xyz.reshape(-1, 1, 3) - xyz.reshape(1, -1, 3)
            distance = (distance - np.rint(distance.astype(float)))
            distance = ((distance @ self.M)**2).sum(axis=2) ** (1/2)
            self.distance = np.triu(distance)
            if step_idx == 0:
                for xrd in self.xrd_list:
                    angsize = self.angle.size
                    self.coords[xrd]['RD'] = np.zeros((steps.size, angsize))
                # diagonal = {xrd: {} for xrd in self.xrd_list}

            N = self.atoms.shape[0]
            # dihedral = 0
            for elem in element:
                if types != None:
                    elem_ = type2elem[elem]
                    label = f"{elem_}-{elem_}"
                else:
                    label = f"{elem}-{elem}"
                num_elems = np.sum(self.elem_types == elem)
                coord = self.calc_XRD(elem, elem, Rc)

                for xrd in self.xrd_list:
                    if xrd == 'XRD':
                        if types != None:
                            f_elem = mp.calc_atom_factor(self.angle,
                                                         self.Lambda,
                                                         elem_)
                        else:
                            f_elem = mp.calc_atom_factor(self.angle,
                                                         self.Lambda,
                                                         elem)
                    else:  # ND
                        f_elem = mp.calc_ND_factor(elem)
                    if step_idx == 0:
                        self.factors[xrd][label] = np.zeros((steps.size,
                                                             angsize))
                        self.coords[xrd][label] = np.zeros((steps.size,
                                                            angsize))
                    factor = 2 * (f_elem ** 2) / N
                    self.factors[xrd][label][step_idx] = factor
                    self.coords[xrd][label][step_idx] = coord
                    # print(factor[self.angle == 29.5],
                    #       coord[self.angle == 29.5])
                    rd = factor * coord + (f_elem ** 2) * num_elems / N
                    # print(((f_elem ** 2) * num_elems / N)[self.angle == 23])
                    self.coords[xrd]['RD'][step_idx, :] += rd.astype(float)
                    # dihedral += (f_elem ** 2) * num_elems / N
            # self.coords[xrd]['RD'][step_idx] += dihedral

            if len(element) > 1:
                for i, elem1 in enumerate(element):
                    if types != None:
                        elem1_ = type2elem[elem1]
                    for j, elem2 in enumerate(element[i+1:]):
                        if types != None:
                            elem2_ = type2elem[elem2]
                            label = f"{elem1_}-{elem2_}"
                        else:
                            label = f"{elem1}-{elem2}"
                        coord = self.calc_XRD(elem1, elem2, Rc)
                        for xrd in self.xrd_list:
                            if xrd == 'XRD':
                                if types != None:
                                    f_elem1 = mp.calc_atom_factor(self.angle,
                                                                  self.Lambda,
                                                                  elem1_)
                                else:
                                    f_elem1 = mp.calc_atom_factor(self.angle,
                                                                  self.Lambda,
                                                                  elem1)
                            else:
                                if types != None:
                                    f_elem1 = mp.calc_ND_factor(elem1_)
                                else:
                                    f_elem1 = mp.calc_ND_factor(elem1)
                            if xrd == 'XRD':
                                if types != None:
                                    f_elem2 = mp.calc_atom_factor(self.angle,
                                                                  self.Lambda,
                                                                  elem2_)
                                else:
                                    f_elem2 = mp.calc_atom_factor(self.angle,
                                                                  self.Lambda,
                                                                  elem2)
                            else:
                                if types != None:
                                    f_elem2 = mp.calc_ND_factor(elem2_)
                                else:
                                    f_elem2 = mp.calc_ND_factor(elem2)
                            if step_idx == 0:
                                self.factors[xrd][label] = np.zeros((steps.size,
                                                                     angsize))
                                self.coords[xrd][label] = np.zeros((steps.size,
                                                                    angsize))
                            coeff = 2 * (f_elem1 * f_elem2) / N
                            self.factors[xrd][label][step_idx] = coeff
                            self.coords[xrd][label][step_idx] = coord
                            # print(coeff[self.angle == 29.5],
                            #       coord[self.angle == 29.5])
                            rd = (coeff * coord).astype(float)
                            self.coords[xrd]['RD'][step_idx] += rd
            step_idx += 1
        # for xrd in self.xrd_list:
        #     self.coords[xrd]['RD'] = self.coords[xrd]['RD'].mean(axis=0)

    def get_ciffile(self, infile, element, step_bin, super_cell):
        with open(infile) as f:
            body = f.read()
        num_cell = super_cell[0] * super_cell[1] * super_cell[2]
          
        # box
        lengths = [float(l) for l in
                   re.findall("_cell_length_[a-c]\s+([\d\.]+)", body)]
        lengths = [l*super_cell[i] for i, l in enumerate(lengths)]
        angles = [float(a) for a in
                  re.findall("_cell_angle_\w+\s+([\d\.]+)", body)]
        self.M = mp.CalcMatrix(*lengths, *angles)
        self.M_ = np.linalg.inv(self.M)

        key_list = re.findall("_atom_site_(.+)", body)
        site = re.findall(f"_site_{key_list[-1]}\n(.+)", body, re.DOTALL)[0]
        site = np.array(site.split(), dtype=object).reshape(-1, len(key_list))
        s_site = np.tile(site, (num_cell, 1))
        types = s_site[:, key_list.index("type_symbol")]
        xyz_idx = [key_list.index(f"fract_{axis}") for axis in ["x", "y", "z"]]

        lx = np.arange(super_cell[0])
        ly = np.arange(super_cell[1])
        lz = np.arange(super_cell[2])
        mx, my, mz = [mesh.reshape(-1, 1) for mesh in np.meshgrid(lx, ly, lz)]
        mesh = np.repeat(np.hstack((mx, my, mz)), site.shape[0], axis=0)
        frac_xyz = s_site[:, xyz_idx].astype(float)
        s_site[:, xyz_idx] = (frac_xyz + mesh) / super_cell

        self.atoms = np.empty((s_site.shape[0], 6), dtype=object)
        self.atoms[:, 0] = np.arange(s_site.shape[0])  # id
        self.atoms[:, 1] = [ELEMENT.index(t)+1 for t in types]  # type
        self.atoms[:, 2] = types.copy()  # element
        self.atoms[:, 3:] = frac_xyz @ self.M
        
        # elem types
        self.elem_types = types.copy()
        elem_idxs = np.empty(0, dtype=int)
        for elem in element:
            elem_id = np.where(self.elem_types == elem)[0]
            elem_idxs = np.append(elem_idxs, elem_id)
        self.elem_types = self.elem_types[elem_idxs]
        frac_xyz = frac_xyz[elem_idxs, :]

        distance = frac_xyz.reshape(-1, 1, 3) - frac_xyz.reshape(1, -1, 3)
        distance = (distance - np.rint(distance.astype(float)))
        distance = ((distance @ self.M)**2).sum(axis=2) ** (1/2)
        self.distance = np.triu(distance)
        for xrd in self.xrd_list:
            angsize = self.angle.size
            self.coords[xrd]['RD'] = np.zeros((1, angsize))

        N = self.atoms.shape[0]
        for elem in element:
            label = f"{elem}-{elem}"
            num_elems = np.sum(self.elem_types == elem)
            coord = self.calc_XRD(elem, elem, None)
            for xrd in self.xrd_list:
                if xrd == 'XRD':
                    f_elem = mp.calc_atom_factor(self.angle, self.Lambda, elem)
                else:  # ND
                    f_elem = mp.calc_ND_factor(elem)
                self.factors[xrd][label] = np.zeros((1, angsize))
                self.coords[xrd][label] = np.zeros((1, angsize))
                factor = 2 * (f_elem ** 2) / N
                self.factors[xrd][label][0] = factor
                self.coords[xrd][label][0] = coord
                rd = factor * coord + (f_elem ** 2) * num_elems / N
                self.coords[xrd]['RD'][0, :] += rd.astype(float)

        if len(element) > 1:
            for i, elem1 in enumerate(element):
                for j, elem2 in enumerate(element[i+1:]):
                    label = f"{elem1}-{elem2}"
                    coord = self.calc_XRD(elem1, elem2, None)
                    for xrd in self.xrd_list:
                        if xrd == 'XRD':
                            f_elem1 = mp.calc_atom_factor(self.angle,
                                                          self.Lambda, elem1)
                        else:
                            f_elem1 = mp.calc_ND_factor(elem1)
                        if xrd == 'XRD':
                            f_elem2 = mp.calc_atom_factor(self.angle,
                                                          self.Lambda, elem2)
                        else:
                            f_elem2 = mp.calc_ND_factor(elem2)
                        self.factors[xrd][label] = np.zeros((1, angsize))
                        self.coords[xrd][label] = np.zeros((1, angsize))
                        coeff = 2 * (f_elem1 * f_elem2) / N
                        self.factors[xrd][label][0] = coeff
                        self.coords[xrd][label][0] = coord
                        rd = (coeff * coord).astype(float)
                        self.coords[xrd]['RD'][0] += rd

    def calc_XRD(self, elem1, elem2, Rc):
        elem1_idx = np.where(self.elem_types == elem1)[0]
        elem2_idx = np.where(self.elem_types == elem2)[0]
        r = self.distance[elem1_idx, :][:, elem2_idx]
        r = r[r != 0].reshape(1, -1)
        # Rc = ((self.length ** 2).sum()) ** (1/2)
        if Rc == None:
            Rc = (np.diag(self.M) ** 2).sum() ** (1/2)
        Q = mp.calc_Qvalue(self.angle, self.Lambda).reshape(-1, 1)
        pi_Rc = np.array([mp.calc_pi_Rc(Rc)])
        return (mp.func(Q, r) * mp.func(pi_Rc, r)).sum(axis=1).ravel()

    def get_plumed(self, infile):
        with open(infile) as f:
            lines = f.readlines()
        keys = lines[0].split()[3:]
        values = lines[1].split()[1:]
        self.angle = np.unique([int(re.search('\d+', key).group(0))
                                for key in keys])
        for idx, key in enumerate(keys):
            angle = int(re.search('\d+', key).group(0))
            label = re.sub('\d+', '', key)
            value = float(values[idx])
            if label not in self.coords:
                self.coords[label] = np.empty(self.angle.size)
            self.coords[label][self.angle == angle] = value

    def calc_factors(self, element, ND=True):
        N = self.elem_types.size
        for elem in element:
            label = f"{elem}-{elem}"
            if ND == True:
                f_elem = mp.calc_atom_factor(self.angle, self.Lambda, elem)
            else:
                f_elem = mp.calc_ND_factor(elem)
            self.factors[label] = 2 * (f_elem ** 2) / N

        if len(element) > 1:
            for i, elem1 in enumerate(element):
                if ND == True:
                    f_elem1 = mp.calc_atom_factor(self.angle, self.Lambda,
                                                  elem1)
                else:
                    f_elem1 = mp.calc_ND_factor(elem1)
                for j, elem2 in enumerate(element[i+1:]):
                    label = f"{elem1}-{elem2}"
                    if ND == True:
                        f_elem2 = mp.calc_atom_factor(self.angle, self.Lambda,
                                                      elem2)
                    else:
                        f_elem2 = mp.calc_ND_factor(elem2)
                    self.factors[label] = 2 * (f_elem1 * f_elem2) / N

    def get_dat(self, colvar):
        datfile = re.sub('.colvar', '.dat', colvar)
        with open(datfile) as df:
            body = df.read()
        for elem, indexs in re.findall('(.+): GROUP ATOMS=([\d\-]+)', body):
            init, last = [int(index) for index in indexs.split('-')]
            elem_type = np.tile(elem, last-init)
            self.elem_types = np.append(self.elem_types, elem_type)

def plot_xrd(x_dict, y_dict, coeff, xrd_list, base, c_map="summer",
             save_fig=True, lines=[22, 33]):
    name_list = list(x_dict.keys())
    if len(xrd_list) > 1:
        label_list = []
        for name in name_list:
            label_list += [f'{name}({xrd})' for xrd in xrd_list]
    else:
        label_list = name_list

    fig_num = len(y_dict[name_list[0]][xrd_list[0]])
    col_num = int(np.ceil(np.sqrt(fig_num)))
    row_num = int(np.ceil(fig_num / col_num))
    fig = rgp.MakeFig(col_num, row_num)
    ax = []

    # y_lim = [0, 0]
    for name_idx, name in enumerate(name_list):
        X = x_dict[name]
        for xrd_idx, xrd in enumerate(xrd_list):
            color_idx = name_idx * len(xrd_list) + xrd_idx

            label = label_list[color_idx]
            for fig_idx, title in enumerate(y_dict[name][xrd].keys()):
                if color_idx == 0:
                    ax.append(fig.add_subplot(col_num, row_num, fig_idx+1))
                    ax[fig_idx].set_xlabel('Angle 2θ/°')
                    if title == "RD":
                        # ax[fig_idx].set_title("total")
                        y_label = "Total"
                    else:
                        # ax[fig_idx].set_title(title)
                        y_label = f"Partial, {title}"
                    ax[fig_idx].set_ylabel(y_label)

                # print(y_dict[name][xrd][title])
                y = y_dict[name][xrd][title].copy()
                if title in factors[name][xrd]:
                    y *= factors[name][xrd][title]
                y = y.mean(axis=0)
                if c_map == 'jet':
                    color = cm.jet(1-(color_idx/len(label_list)))
                elif c_map == 'summer':
                    color = cm.summer(color_idx/len(label_list))
                elif c_map == 'cool':
                    color = cm.cool(color_idx/len(label_list))
                if label == "acc":
                    ax[fig_idx].plot(X, y, label=label, c=color, p="*")
                else:
                    ax[fig_idx].plot(X, y, label=label, c=color)
                
                # if title == "RD":
                #     if y_lim[0] > y.min():
                #         y_lim[0] = y.min()
                #     if y_lim[1] > y.max():
                #         y_lim[1] = y.max()

    RD_idx = list(y_dict[name][xrd].keys()).index("RD")  # fig_idx
    for ang_idx, angle in enumerate(lines):
        x_list = [angle, angle]
        y_list = ax[RD_idx].get_ylim()
        # ax[RD_idx].plot(x_list, y_list, "g--", linewidth=4)
        x_loc = angle + 1
        y_loc = y_list[1] * 0.05 + y_list[0] * 0.95
        # ax[RD_idx].text(x_loc, y_loc, f"CV{ang_idx+1}", color="g",
        #                 fontsize=16, ha='left', va='bottom')
    for ax_ in ax:
        ax_.legend(loc=(0.55, 0.6), frameon=False)
    plt.show()

    if save_fig == True:
        fig.patch.set_alpha(0)  # transparent=True -> all
        fig.savefig(f"{base}.pdf")
        print(f"{base}.pdf created")
    
if __name__ == "__main__":
    import argparse

    par = argparse.ArgumentParser(description="bar")
    par.add_argument('infiles', help='lammpstrj file', nargs="+")
    par.add_argument('-a', '--atoms', nargs="+", default=None)
    par.add_argument('-e', '--element', nargs="+", default=['C', 'Ca'])
    par.add_argument('-t', '--types', nargs="+", type=int, default=None)
    par.add_argument('-l', '--lambda_', type=float, default=1.54059,
                     help="wavelength[Ang]")
    par.add_argument('-s', '--step_bin', type=int, default=10,
                     help="step bin[default=1]")
    par.add_argument('-c', '--c_map', choices=['jet', 'summer', 'cool'],
                     default='cool')
    par.add_argument('-sc', '--super_cell', type=int, nargs=3,
                     default=[3, 2, 1],
                     help="super cell(only ciffile)")
    par.add_argument('-o', '--outfile', help='output name')
    par.add_argument('-f', '--factor', action="store_true", help="default=on")
    par.add_argument('-x', '--xrd', choices=['XRD', 'ND', 'COMP'],
                     default='XRD', help='default : xrd')
    par.add_argument('-rc', '--Rc', type=float, default=None)
    args = par.parse_args()

    if args.atoms != None:
        ELEMENT = args.atoms
    # angle_num = 351
    angle_num = 351

    angles = {}
    coords = {}
    factors = {}
    for infile in tqdm(args.infiles):
        label, ext = os.path.splitext(os.path.basename(infile))
        # label = remove_labelname(label)
        xrd = XRD(args.lambda_, args.xrd)
        if ext == '.lammpstrj':
            # label += '(calc)'
            xrd.angle = np.linspace(15, 50, angle_num)
            xrd.get_trjfile(infile, args.element, args.step_bin,
                            types=args.types, Rc=args.Rc)
        elif ext == '.cif':
            # label += '(calc)'
            xrd.angle = np.linspace(15, 50, angle_num)
            xrd.get_ciffile(infile, args.element, args.step_bin,
                            args.super_cell)
        elif args.factor != True:
            label += '(plumed)'
            xrd.get_plumed(infile)
            xrd.get_dat(infile)
            xrd.calc_factors(args.element)
        else:
            label += '(plumed)'
            xrd.get_dat(infile)
        angles[label] = xrd.angle
        coords[label] = xrd.coords
        xrd_list = xrd.xrd_list
        if args.factor != True:
            factors[label] = xrd.factors
        else:
            factors[label] = {xrd: {} for xrd in xrd_list}



    if args.outfile == None:
        base = 'XRD'
    else:
        base = args.outfile

    plot_xrd(angles, coords, factors, xrd_list, base, c_map=args.c_map)
    # ax.set_ylim(-100, 200)
    # if args.outfile:
    #     plt.savefig(args.outfile, dpi=300)
    # else:
    #     plt.show()
