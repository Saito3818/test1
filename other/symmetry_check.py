#!/usr/bin/env python

from spglib import get_symmetry_dataset
import re
import os
import numpy as np
import tqdm

def CalcMatrix(a, b, c, alpha, beta, gamma):
    alpha = alpha * np.pi/180
    beta = beta * np.pi/180
    gamma = gamma * np.pi/180
    v1 = [a, 0, 0]
    v2 = [b*np.cos(gamma),  b*np.sin(gamma), 0]
    v3 = [c*np.cos(beta),
          c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
          c*np.sqrt(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                    - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2) /
          np.sin(gamma)]
    return np.array([v1, v2, v3])


class SYMMETRY():
    def __init__(self, symprec=1e-1):
        self.a = []
        self.symprec = symprec  # Ang.
        # base = os.path.splitext(os.path.basename(trjfile)[0])

    def read_trjfile(self, trjfile):
        with open(trjfile) as o:
            body = o.read()
        times = re.findall('ITEM: TIMESTEP\n(.+?)\nITEM', body, re.DOTALL)
        boxs = re.findall('ITEM: BOX.+?\n(.+?)\nITEM', body, re.DOTALL)
        boxs = np.array([box.split() for box in boxs], dtype=float)
        if boxs.shape[1] == 6:
            boxs = boxs.reshape(-1, 3, 2)
            triangle = False
        elif boxs.shape[1] == 9:
            boxs = boxs.reshape(-1, 3, 3)
            triangle = True
        key = re.findall('ITEM: ATOMS (.+?\n)', body)[0].split()
        type_idx = key.index('type')
        xyz_idx = [key.index('x'), key.index('y'), key.index('z')]
        num_atoms = int(re.findall('ITEM: NUMBER .+?\n(.+?)\n', body)[0])
        atoms = re.findall('ITEM: ATOMS.+?\n(.+?)\nITEM', body, re.DOTALL)
        atoms.append(re.findall('.+ITEM: ATOMS.+?\n(.+)', body, re.DOTALL)[0])
        atoms = np.array([atom.split() for atom in atoms], dtype=object)
        atoms = atoms.reshape(-1, num_atoms, len(key))
        for idx in range(atoms.shape[0]):
            atom = atoms[idx, :]
            box = boxs[idx, :, :]
            if triangle:
                xlo_bound, xhi_bound = box[0, 0], box[0, 1]
                ylo_bound, yhi_bound = box[1, 0], box[1, 1]
                zlo_bound, zhi_bound = box[2, 0], box[2, 1]
                xy, xz, yz = 0.0, 0.0, 0.0
                xy, xz, yz = box[0, 2], box[1, 2], box[2, 2]
                xlo = xlo_bound - np.min([0.0, xy, xz, xy+xz])
                xhi = xhi_bound - np.max([0.0, xy, xz, xy+xz])
                ylo = ylo_bound - np.min([0.0, yz])
                yhi = yhi_bound - np.max([0.0, yz])
                lx, ly, lz = xhi - xlo, yhi - ylo, zhi_bound - zlo_bound
                a = lx
                b = np.sqrt(ly**2 + xy**2)
                c = np.sqrt(lz**2 + xz**2 + yz**2)
                alpha = np.arccos((xy*xz + ly*yz)/(b*c)) * 180/np.pi
                beta = np.arccos(xz/c) * 180/np.pi
                gamma = np.arccos(xy/b) * 180/np.pi
                lattice = CalcMatrix(a, b, c, alpha, beta, gamma)
            else:
                lattice = [[box[0, 1]-box[0, 0], 0, 0],
                           [0, box[1, 1]-box[1, 0], 0],
                           [0, 0, box[2, 1]-box[2, 0]]]
                # length = lengths[idx, :]
            position = np.dot(atom[:, xyz_idx].astype(float),
                              np.linalg.inv(lattice))  # fractional coord
            types = atom[:, type_idx].astype(int)
            cell = (lattice, position, types)
            dataset = get_symmetry_dataset(cell, symprec=self.symprec)
            print(f"{times[idx]}step: {dataset['international']}: {dataset['number']}")
            # print(dataset)

        # print(dataset['international'])
        return cell
        
    def SetMatrix(self, length, angle):
        alpha = angle[0] * np.pi/180
        beta = angle[1] * np.pi/180
        gamma = angle[2] * np.pi/180

        a, b, c = length
        v1 = [a, 0, 0]
        v2 = [b*np.cos(gamma),  b*np.sin(gamma), 0]
        v3 = [c*np.cos(beta),
              c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
              c*np.sqrt(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                        - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2) /
              np.sin(gamma)]
        self.lattice = np.array([v1, v2, v3])
        self.lattice_ = np.linalg.inv(self.lattice)
        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = np.sqrt(b**2 - xy**2)
        yz = (b * c * np.cos(alpha) - xy * xz)/ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)
        lammps_lattice = [[0, lx, xy], [0, ly, xz], [0, lz, yz]]
        self.lammps_lattice = np.round(np.array(lammps_lattice), 6)

    def convert_fig(self, figures):
        return [float(re.sub('[\(\)]', '', figure)) for figure in figures]

    def read_ciffile(self, ciffile):
        with open(ciffile) as o:
            body = o.read()
        length = [float(l) for l in
                  re.findall('_cell_length_.+?([\.\d]+)', body)]
        angle = [float(l) for l in
                 re.findall('_cell_angle.+?([\.\d]+)', body)]
        self.SetMatrix(length, angle)

        site_key = re.findall('_atom_site_(.+)', body)
        site_key = [sk for sk in site_key if re.search('aniso', sk) == None]
        try:
            site = re.findall(f'{site_key[-1]}\n(.+)\n\n', body, re.DOTALL)[0]
        except IndexError:
            site = re.findall(f'{site_key[-1]}\n(.+)', body, re.DOTALL)[0]
        site_key = [re.sub("\s", "", sk) for sk in site_key]
        
        site = np.array(site.split(), dtype=object).reshape(-1, len(site_key))
        pos_key = [site_key.index('fract_x'), site_key.index('fract_y'),
                   site_key.index('fract_z')]
        position = np.array([self.convert_fig(s) for s in site[:, pos_key]])
        type_symbol = site[:, site_key.index('type_symbol')]
        type_idx = {ts: idx+1 for idx, ts in enumerate(np.unique(type_symbol))}
        types = [type_idx[ts] for ts in type_symbol]
        return (self.lattice, position, types)

    def read_xdatcar(self, infile):
        with open(infile) as o:
            body = o.read()

        header = re.search(".+\n", body).group(0)
        unit = re.findall(f"{header}(.+?)\nDirect configu", body, re.DOTALL)
        pos = re.findall("Direct configu.+?\n([\s\d\.\-]+)", body, re.DOTALL)
        if len(unit) == len(pos):
            NPT_TRUE = True
        else:
            NPT_TRUE = False  # NVT
            lattice = np.array(unit[0].split()[1:10], dtype=float)
            lattice = lattice.reshape(3, 3)
            
            # ly = float(unit_data[5])
            # lz = float(unit_data[9])
            # xy = float(unit_data[4])
            # xz = float(unit_data[7])
            # yz = float(unit_data[8])
            # a = float(unit_data[1])
            # b = np.sqrt(ly**2 + xy**2)
            # c = np.sqrt(lz**2 + xz**2 + yz**2)
            # alpha = np.arccos((xy*xz + ly*yz)/(b*c)) * 180/np.pi
            # beta = np.arccos(xz/c) * 180/np.pi
            # gamma = np.arccos(xy/b) * 180/np.pi
            # lattice = CalcMatrix(a, b, c, alpha, beta, gamma)
            

        type_data = unit[0].split()[10:]
        type_num = len(type_data) // 2  # to int
        type_idx = np.array(type_data[type_num:], dtype=int)
        types = np.empty(type_idx.sum(), dtype=int)
        t_n_init = 0
        for idx, t_n in enumerate(type_idx):
            types[t_n_init:t_n_init+t_n] = idx + 1
            t_n_init += t_n
        # type_dict = {type_data[idx]: int(type_data[idx+type_num])
        #              for idx in range(type_num)}
        
        for idx, position in enumerate(pos):
            if NPT_TRUE:
                lattice = np.array(unit[idx].split()[1:10], dtype=float)
                lattice = lattice.reshape(3, 3)
            position = np.array(position.split(), dtype=float).reshape(-1, 3)
            cell = (lattice, position, types)
            dataset = get_symmetry_dataset(cell, symprec=self.symprec)
            print(f"{idx}step: {dataset['international']}: {dataset['number']}")

    def read_xdatcar(self, infile):
        with open(infile) as o:
            body = o.read()
        body = body.split("\n\n")[0].split("(.+)\nDirect")
        unit = body[0].split()
        lattice = np.array(unit[2:11], dtype=float).reshape(3, 3)
        type_num = (len(unit) - 10) // 2  # to int
        type_idx = np.array(unit[-type_num:], dtype=int)
        types = np.empty(type_idx.sum(), dtype=int)
        t_n_init = 0
        for idx, t_n in enumerate(type_idx):
            types[t_n_init:t_n_init+t_n] = idx + 1
            t_n_init += t_n

        position = np.array(body.split(), dtype=float).reshape(-1, 3)
        cell = (lattice, position, types)
        dataset = get_symmetry_dataset(cell, symprec=self.symprec)
        print(f"{dataset['international']}: {dataset['number']}")
            
    
if __name__ == '__main__':
    import argparse

    par = argparse.ArgumentParser(description="test")
    par.add_argument('trjfiles', nargs='+')
    par.add_argument('-s', '--symprec', type=float, default=5e-1)
    # par.add_argument('-n', '--nolegend', default=True, action='store_false')
    args = par.parse_args()


    for trjfile in args.trjfiles:
        sm = SYMMETRY(symprec=args.symprec)
        base, ext = os.path.splitext(os.path.basename(trjfile))
        if ext == '.lammpstrj':
            sm.read_trjfile(trjfile)
        elif ext == '.cif':
            cell = sm.read_ciffile(trjfile)
            dataset = get_symmetry_dataset(cell, symprec=args.symprec)
            
            # print(f"{dataset['international']}: {dataset['number']}")
            print(dataset["std_lattice"])
            # print(dataset["rotations"], dataset["translations"])        
        elif ext == "":
            if base == "XDATCAR":
                cell = sm.read_xdatcar(trjfile)
            elif base == "POSCAR":
                cell = sm.read_poscar(trjfile)
        else:
            print(f"base: {base} ext: {ext} file not accept")
