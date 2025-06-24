#!/usr/bin/env python
import re
import os
import numpy as np

MASS = {'40.08': 'Ca', '12.01': 'C', '16.00': 'O', '1.01': 'H', '28.0855': 'Si'}
SCATTER = {}

# H
SCATTER['H'] = {}
SCATTER['H']['A'] = [4.130480e-01, 2.949530e-01, 1.874910e-01, 8.070100e-02,
                     2.373600e-02]
SCATTER['H']['B'] = [1.556995e+01, 3.239847e+01, 5.711404e+00, 6.188987e+01,
                     1.334118e+00]
SCATTER['H']['C'] = 4.900000e-05
SCATTER['H']['nc'] = -3.7390
# C
SCATTER['C'] = {}
SCATTER['C']['A'] = [2.657506e+00, 1.078079e+00, 1.490909e+00, -4.241070e+00,
                     7.137910e-01]
SCATTER['C']['B'] = [1.478076e+01, 7.767750e-01, 4.208684e+01, -2.940000e-04,
                     2.395350e-01]
SCATTER['C']['C'] = 4.297983e+00
SCATTER['C']['nc'] = 6.6460
# O
SCATTER['O'] = {}
SCATTER['O']['A'] = [2.960427e+00, 2.508818e+00, 6.378530e-01, 7.228380e-01,
                     1.142756e+00]
SCATTER['O']['B'] = [1.418226e+01, 5.936858e+00, 1.127260e-01, 3.495848e+01,
                     3.902400e-01]
SCATTER['O']['C'] = 2.701400e-02
SCATTER['O']['nc'] = 5.803
# Si
SCATTER['Si'] = {}
SCATTER['Si']['A'] = [5.275329e+00, 3.191038e+00, 1.511514e+00, 1.356849e+00,
                      2.519114e+00]
SCATTER['Si']['B'] = [2.631338e+00, 3.373073e+01, 8.111900e-02, 8.628864e+01,
                      1.170087e+00]
SCATTER['Si']['C'] = 1.450730e-01
SCATTER['Si']['nc'] = 4.1491
# Ca
SCATTER['Ca'] = {}
SCATTER['Ca']['A'] = [8.593655e+00, 1.477324e+00, 1.436254e+00, 1.182839e+00,
                      7.113258e+00]
SCATTER['Ca']['B'] = [1.046064e+01, 4.189100e-02, 8.139038e+01, 1.698478e+02,
                      6.880980e-01]
SCATTER['Ca']['C'] = 1.962550e-01
SCATTER['Ca']['nc'] = 4.70

def calc_Qvalue(angle, lambda_):
    """angle: 2 theta
       lambda_: wavelength of the radiation
    """
    return 4 * np.pi / lambda_ * np.sin(angle * np.pi / 360)

def calc_pi_Rc(Rc):  # Window Func.
    return np.pi / Rc

def func(coeff, r):
    # value = np.zeros((coeff.shape[0], r.shape[1]))
    # arg_ = (coeff * r[r != 0]).astype(float)
    # value[:, np.where(r != 0)[1]] = np.sin(arg_) / arg_
    # return value
    return np.sin((coeff * r).astype(float)) / (coeff * r)


def calc_atom_factor(angle, lambda_, elem):
    """calculation atomic scattering form factors
    """
    q = (np.sin(angle * np.pi / 360) / lambda_) ** 2
    factor = SCATTER[elem]['C']
    for sct_idx, A in enumerate(SCATTER[elem]['A']):
        B = SCATTER[elem]['B'][sct_idx]
        factor += A * np.exp(-B * q)
    return factor

def calc_ND_factor(elem):
    """calculation atomic scattering form factors
    """
    # q = (np.sin(angle * np.pi / 360) / lambda_) ** 2
    factor = SCATTER[elem]['nc']

    return factor

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


class PLIUMED():
    def __init__(self, lmp_data):
        self.lmp_data = []
        self.lmp_idxs = {}

        self.atoms = np.empty(0)
        self.velcs = np.empty(0)
        self.atom_idxs = np.empty(0)  # idx:new_idx value:old_idx
        self.type_idxs = {}

        self.read_lmp_data(lmp_data)
        # self.make_plumed_data(element, angle)

    def read_lmp_data(self, lmp_data):
        with open(lmp_data) as o:
            body = o.read()
        self.lmp_data = body.split('\n\n')
        body_types = ['Masses', 'Bond Coeffs', 'Atoms', 'Velocities',
                      'Bonds', 'Angles', 'Impropers']
        for lmp_idx, data in enumerate(self.lmp_data):
            for body_type in body_types:
                if re.search(body_type, data) != None:
                    self.lmp_idxs[body_type] = lmp_idx + 1

        try:
            num_atoms = int(re.findall('(\d+) atoms', self.lmp_data[1])[0])
            num_types = int(re.findall('(\d+) atom types', self.lmp_data[1])[0])
        except IndexError:
            num_atoms = int(re.findall('(\d+) atoms', self.lmp_data[0])[0])
            num_types = int(re.findall('(\d+) atom types', self.lmp_data[0])[0])

        box = np.array(self.lmp_data[self.lmp_idxs['Masses']-2].split())
        if box.size == 12:
            box = box.astype(object).reshape(3, 4)
            box[:, :2] = box[:, :2].astype(float)
            length = (box[:3, 1] - box[:3, 0]).ravel()
            self.M = CalcMatrix(length[0], length[1], length[2], 90, 90, 90)
        else:  # triclinic
            xy, xz, yz = [float(b) for b in box[12:15]]
            box = box[:12].astype(object).reshape(3, 4)
            box[:, :2] = box[:, :2].astype(float)
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
            self.M = CalcMatrix(a, b, c, alpha, beta, gamma)

        atoms = np.array(self.lmp_data[self.lmp_idxs['Atoms']].split())
        self.atoms = atoms.astype(object).reshape(num_atoms, -1) 
        self.atoms[:, :3] = self.atoms[:, :3].astype(int)
        self.atoms[:, 3:7] = self.atoms[:, 3:7].astype(float)
        # self.atoms[:, 7:] = self.atoms[:, 7:].astype(int)
        self.atoms = self.atoms[self.atoms[:, 0].argsort(), :]

        self.atom_idxs = np.empty(0).astype(int)
        for atom_type in range(1, num_types+1):
            atom_idx = np.where(self.atoms[:, 2] == atom_type)[0]
            self.atom_idxs = np.append(self.atom_idxs, atom_idx)
            self.type_idxs[str(atom_type)] = atom_idx.size
        self.atoms = self.atoms[self.atom_idxs, :]
        self.atoms[:, 0] = np.arange(1, num_atoms+1).astype(int)

        if 'Velocities' in self.lmp_idxs:
            velcs = np.array(self.lmp_data[self.lmp_idxs['Velocities']].split())
            self.velcs = velcs.astype(object).reshape(num_atoms, -1)
            self.velcs[:, 0] = self.velcs[:, 0].astype(int)
            self.velcs[:, 1:] = self.velcs[:, 1:].astype(float)
            self.velcs = self.velcs[self.velcs[:, 0].argsort(), :]
            self.velcs = self.velcs[self.atom_idxs, :]
            self.velcs[:, 0] = np.arange(1, num_atoms+1).astype(int)

    def make_plumed_data(self, element, angles, lambda_, base, height, ND, Rc):
        outfile = f'{base}.dat'
        outbody = "UNITS ENERGY=kcal/mol LENGTH=A TIME=fs\n"

        outbody += "# metadynamics using XRD peak CV\n"
        masses = np.array([mass.split() for mass in
                           self.lmp_data[self.lmp_idxs['Masses']].split('\n')])
        masses = masses.astype(object)[:, :2]
        atom_idx = 1

        num_elems = {}
        for atom_type, mass in masses:
            symbol = MASS[f'{round(float(mass), 2):.2f}']
            type_idx = self.type_idxs[atom_type]
            num_elems[symbol] = type_idx
            outbody += f"{symbol}: GROUP "
            outbody += f"ATOMS={atom_idx:d}-{type_idx+atom_idx-1:d}\n"
            atom_idx += type_idx

        if Rc == None:
            Rc = (np.diag(self.M) ** 2).sum() ** (1/2)  # * 0.85526
        # print("Rc:", Rc)
        pi_Rc = calc_pi_Rc(Rc)
        N = self.atoms.shape[0]
        all_ARG_list = 'ARG='
        for angle_ in angles:
            outbody += f"# {angle_}\n"
            Q = calc_Qvalue(angle_, lambda_)
            angle = int(angle_)
            func = f"FUNC=sin({Q:.6f}*x)/{Q:.6f}/x"
            func += f"*sin({pi_Rc:.6f}*x)/{pi_Rc:.6f}/x"
            switch = f"SWITCH={{CUSTOM {func} R_0=1}}"
            arg_list = {}

            diagonal = 0
            for elem in element:
                outbody += f"{elem}{elem}{angle}: COORDINATION "
                outbody += f"GROUPA={elem} {switch}\n"
                if ND != True:
                    f_elem = calc_atom_factor(angle_, lambda_, elem)
                else:
                    f_elem = calc_ND_factor(elem)
                # arg_list[f"{elem}{elem}{angle}"] = 2 * (f_elem ** 2) / N
                arg_list[f"{elem}{elem}{angle}"] = 2 * (f_elem ** 2) / N
                diagonal += (f_elem ** 2) * num_elems[elem] / N
            if len(element) > 1:
                for i, elem1 in enumerate(element):
                    if ND != True:
                        f_elem1 = calc_atom_factor(angle_, lambda_, elem1)
                    else:
                        f_elem1 = calc_ND_factor(elem1)
                    for j, elem2 in enumerate(element[i+1:]):
                        outbody += f"{elem1}{elem2}{angle}: COORDINATION "
                        outbody += f"GROUPA={elem1} GROUPB={elem2} {switch}\n"
                        if ND != True:
                            f_elem2 = calc_atom_factor(angle_, lambda_, elem2)
                        else:
                            f_elem2 = calc_ND_factor(elem2)
                        coeff = 2 * f_elem1 * f_elem2 / N
                        arg_list[f"{elem1}{elem2}{angle}"] = coeff

            ARG_list = ','.join(list(arg_list.keys()))
            ARG = 'ARG=' + ARG_list
            var_list = []
            func_list = []

            for var_idx, coeff in enumerate(arg_list.values()):
                var = chr(97+var_idx)
                func = f'{var}*{coeff:.6f}'
                var_list.append(var)
                func_list.append(func)
            VAR = 'VAR=' + ','.join(var_list)
            FUNC = 'FUNC=(' + '+'.join(func_list) + f'+{diagonal:.4f})'
            outbody += f'RD{angle}: CUSTOM {ARG} {VAR} {FUNC} PERIODIC=NO\n'
            all_ARG_list += f'RD{angle},{ARG_list},'

            outbody += f'wall: LOWER_WALLS ARG=RD{angle} AT=-150 KAPPA=100\n'
            outbody += f'metad: METAD ARG=RD{angle} ...\n'
            outbody += f'PACE=1000 HEIGHT={height} SIGMA=5 FILE=HILLS\n'
            outbody += 'GRID_MIN=-200 GRID_MAX=400 #GRID_BIN=7000\n'
            outbody += '...\n'
        outbody += f'PRINT {all_ARG_list}metad.* '
        outbody += 'FILE=COLVAR STRIDE=200\n\n'
        outbody += 'ENDPLUMED'

        with open(outfile, 'w') as w:
            w.write(outbody)
        print(f"{outfile} was created")

    def write_lmp_data(self, base):
        outfile = f"{base}.data"
        outbody = ""
        conv_idxs = np.argsort(self.atom_idxs)
        # print(self.lmp_idxs.keys())
        for lmp_idx, data in enumerate(self.lmp_data):
            if lmp_idx == self.lmp_idxs['Atoms']:
                for atom in self.atoms:
                    outbody += " ".join(atom.astype(str)) + '\n'
                outbody += '\n'
                continue
            if 'Velocities' in self.lmp_idxs:
                if lmp_idx == self.lmp_idxs['Velocities']:
                    for velc in self.velcs:
                        outbody += " ".join(velc.astype(str)) + '\n'
                    outbody += '\n'
                    continue
            if ('Bonds' in self.lmp_idxs) & (lmp_idx
                                             == self.lmp_idxs['Bonds']):
                try:
                    bonds = np.array(data.split(), dtype=int)
                except ValueError:  # コメントアウト処理
                    bonds = np.array(re.findall('([\d]+)', data), dtype=int)
                bonds = bonds.reshape(-1, 4)
                bonds[:, 2:] = conv_idxs[bonds[:, 2:] - 1] + 1
                for bond in bonds:
                    outbody += " ".join(bond.astype(str)) + '\n'
                outbody += '\n'
                continue
            if ('Angles' in self.lmp_idxs) & (lmp_idx
                                              == self.lmp_idxs['Angles']):
                try:
                    angles = np.array(data.split(), dtype=int)
                except ValueError:
                    angles = np.array(re.findall('(\d+)', data), dtype=int)
                angles = angles.reshape(-1, 5)
                angles[:, 2:] = conv_idxs[angles[:, 2:] - 1] + 1
                for angle in angles:
                    outbody += " ".join(angle.astype(str)) + '\n'
                outbody += '\n'
                continue
            if ('Impropers' in self.lmp_idxs) & (lmp_idx
                                                 == self.lmp_idxs['Impropers']):
                try:
                    impropers = np.array(data.split(), dtype=int)
                except ValueError:
                    impropers = np.array(re.findall('(\d+)', data), dtype=int)
                impropers = impropers.reshape(-1, 6)
                impropers[:, 2:] = conv_idxs[impropers[:, 2:] - 1] + 1
                for impr in impropers:
                    outbody += " ".join(impr.astype(str)) + '\n'
                continue
            if 'Bond Coeffs' in self.lmp_idxs:
                coeff_idx = self.lmp_idxs['Bond Coeffs']
                if (lmp_idx == coeff_idx-1) | (lmp_idx == coeff_idx):
                    continue
                # else:
                #     outbody += data + '\n\n'
            outbody += data + '\n\n'
        with open(outfile, 'w') as w:
            w.write(outbody)
        print(f"{outfile} was created")

    def write_lmp_in(self, base, infile):
        outfile = f"{base}.in"
        with open(infile) as o:
            body = o.read()
        body = re.sub('\S+\.log', f'{base}.log', body)
        body = re.sub('write.+\n', '', body)
        body = re.sub('\S+\.data', f'{base}.data', body)
        body = re.sub('\S+\.lammpstrj', f'{base}.lammpstrj', body)

        fix = f"fix                 10 all plumed plumedfile {base}.dat "
        fix += f"outfile {base}_plumed.log"
        body = re.sub('\nrun', f'\n{fix}\nrun', body)
        with open(outfile, 'w') as w:
            w.write(body)
        print(f"{outfile} was created")

if __name__ == '__main__':
    import argparse
    
    par = argparse.ArgumentParser(description="test")
    par.add_argument('datafile', help='lammps data file')
    par.add_argument('-i', '--infile', default=None,
                     help='lammps in file')
    par.add_argument('-e', '--element', nargs='+', default=['C', 'Ca', 'O'])
    par.add_argument('-a', '--angle', type=float, nargs='+', default=[30],
                     help="2theta[degree]")
    par.add_argument('-l', '--lambda_', type=float, default=1.54059,
                     help="wavelength[Ang]")
    par.add_argument('-ht', '--height', type=float, default=5)
    par.add_argument('-b', '--base', default='acc_300')
    par.add_argument('-n', '--nd', action='store_true',
                     help='False: xrd, True:nd')
    par.add_argument('-rc', '--Rc', type=float, default=None)
    # par.add_argument('-n', '--nolegend', default=True, action='store_false')
    args = par.parse_args()

    pl = PLIUMED(args.datafile)
    pl.write_lmp_data(args.base)
    pl.make_plumed_data(args.element, args.angle, args.lambda_, args.base,
                        args.height, ND=args.nd, Rc=args.Rc)
    if args.infile != None:
        pl.write_lmp_in(args.base, args.infile)
