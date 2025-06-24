#!/usr/bin/env python
import numpy as np
import argparse
import os


head = """data_VESTA_phase_1

_chemical_name_common                  '{base}'
_cell_length_a                         {a}
_cell_length_b                         {b}
_cell_length_c                         {c}
_cell_angle_alpha                      {alpha}
_cell_angle_beta                       {beta}
_cell_angle_gamma                      {gamma}
_space_group_name_H-M_alt              'P 1'
_space_group_IT_number                 1

loop_
_space_group_symop_operation_xyz
   'x, y, z'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
"""
fmt = "{:5s}{:7.1f}{:13.6f}{:13.6f}{:13.6f} Biso 1.0 {}\n"


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

def dump2cif(trjfile, step=None):
    lines = open(trjfile).readlines()
    atoms = int(lines[3])
    steps = [int(line) for line in lines[1::atoms+9]]
    if step != None:
        step_idx = steps.index(step)
    else:
        step = steps[-1]
        step_idx = len(steps) - 1
    # steps = int(len(lines)/(atoms + 9))
    data = np.array(lines).reshape((len(steps), (atoms + 9)))
    data = data[step_idx, :]

    L = np.array(" ".join(data[5:8]).split(), dtype=float).reshape((3, -1))
    xlo_bound, xhi_bound = L[0, 0], L[0, 1]
    ylo_bound, yhi_bound = L[1, 0], L[1, 1]
    zlo_bound, zhi_bound = L[2, 0], L[2, 1]
    xy, xz, yz = 0.0, 0.0, 0.0
    if L.shape[1] == 3:
        xy, xz, yz = L[0, 2], L[1, 2], L[2, 2]
    xlo = xlo_bound - np.min([0.0, xy, xz, xy+xz])
    xhi = xhi_bound - np.max([0.0, xy, xz, xy+xz])
    ylo = ylo_bound - np.min([0.0, yz])
    yhi = yhi_bound - np.max([0.0, yz])
    lx, ly, lz = xhi - xlo, yhi - ylo, zhi_bound - zlo_bound
    a, b, c = lx, np.sqrt(ly**2 + xy**2), np.sqrt(lz**2 + xz**2 + yz**2)
    alpha = np.arccos((xy*xz + ly*yz)/(b*c)) * 180/np.pi
    beta = np.arccos(xz/c) * 180/np.pi
    gamma = np.arccos(xy/b) * 180/np.pi
    M = CalcMatrix(a, b, c, alpha, beta, gamma)
    a, b, c = a, b, c
    alpha, beta, gamma = alpha, beta, gamma
    print("a = {}".format(a))
    print("b = {}".format(b))
    print("c = {}".format(c))
    print("alpha = {}".format(alpha))
    print("beta  = {}".format(beta))
    print("gamma = {}".format(gamma))

    indexs = data[8].split()[2:]
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

    # xyz
    data = np.array(" ".join(data[9:9+atoms]).split(), dtype=object)
    data = data.reshape((atoms, -1))
    elem = data[:, elem_idx]
    ids = np.copy(elem)
    for e in np.unique(elem):
        idx = np.where(ids == e)[0]
        ids[idx] = ids[idx] + np.arange(len(idx)).astype(str)
    xyz = data[:, xyz_idx].astype(float)
    xyz = np.dot(xyz, np.linalg.inv(M))


    body = head.format(base=trjfile,
                       a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    for i, r in enumerate(xyz):
        body += fmt.format(ids[i], 1.0, r[0], r[1], r[2], elem[i])
    return body, step

if __name__ == '__main__':
    par = argparse.ArgumentParser(description="test")
    par.add_argument('trjfile')
    par.add_argument('-s', '--step', default=None, type=int,
                     help="default : -1")
    # par.add_argument('--xyz', nargs=3, default=[3, 4, 5], type=int,
    #                  helt="xyz index, default=[3, 4, 5]")
    # par.add_argument('--elem', default=2, type=int)
    args = par.parse_args()


    base, ext = os.path.splitext(args.trjfile)
    outbody, step = dump2cif(args.trjfile, args.step)
    outfile = base + "_" + str(step) + ".cif"
    o = open(outfile, "w")
    # o.write(head)
    o.write(outbody)
    print(outfile, "was created.")
