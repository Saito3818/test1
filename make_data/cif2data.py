#!/usr/bin/env python
# A script for MD simulation of amorphous carbonate-water system
# https://doi.org/10.1021/jp910977a

import numpy as np
import argparse
import re


CUTOFF = {}  # bond distance
CUTOFF['C-O'] = 1.50   # origin:1.35
CUTOFF['O-H'] = 1.1
MAXCHAIN = 100

# Care the order to define atoms
VDW = {}  # [mass charge]
VDW["Mg"] = [24.310, 2.000000]
VDW["Ca"] = [40.080, 2.000000]
VDW["Sr"] = [87.620, 2.000000]
VDW["Ba"] = [137.330, 2.000000]
VDW["C"] = [12.010, 1.123285]
VDW["O"] = [16.000, -1.041095]
VDW["Ow"] = [16.000, -0.820000]
VDW["Hw"] = [1.010, 0.410000]

ELEMENT_ORDER = ["Mg", "Ca", "Sr", "Ba", "C", "O", "H"]


class LATTICE():
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.alpha = 0
        self.beta = 0
        self.gamma = 0

    def SetMatrix(self):
        alpha = self.alpha * np.pi/180
        beta = self.beta * np.pi/180
        gamma = self.gamma * np.pi/180
        a, b, c = self.a, self.b, self.c
        v1 = [a, 0, 0]
        v2 = [b*np.cos(gamma),  b*np.sin(gamma), 0]
        v3 = [c*np.cos(beta),
              c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
              c*np.sqrt(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                        - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2) /
              np.sin(gamma)]
        self.M = np.array([v1, v2, v3])
        self.M_ = np.linalg.inv(self.M)
        lx = a
        xy = b * np.cos(gamma)
        xz = c * np.cos(beta)
        ly = np.sqrt(b**2 - xy**2)
        yz = (b * c * np.cos(alpha) - xy * xz)/ly
        lz = np.sqrt(c**2 - xz**2 - yz**2)
        lammps_lattice = [[0, lx, xy], [0, ly, xz], [0, lz, yz]]
        self.lammps_lattice = np.round(np.array(lammps_lattice), 6)


class MOLECULE():
    def __init__(self, molname):
        self.molname = molname  # args.ciffile
        self.lattice = LATTICE()  # calc lattice contants
        self.chains = {}
        self.rings = {}
        # 0:id 1:molid 2:type 3:charge 4:x 5:y 6:z 7:element
        # 8:ff_symbol 9:molname
        self.atoms = np.empty((0, 10), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:symbol
        self.bonds = np.empty((0, 5), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:symbol
        self.angles = np.empty((0, 6), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:id_d 6:symbol
        self.dihedrals = np.empty((0, 7), dtype=object)
        # 0:id 1:type 2:id_a 3:id_b 4:id_c 5:id_d 6:symbol
        self.impropers = np.empty((0, 7), dtype=object)

        # 0:type 1:symbol 2:ff 3:mass 4:charge
        # 5:func 6:eps 7:sigma 8:elem 9:molname
        self.atom_coeffs = np.empty((0, 10), dtype=object)
        # 0:type 1:func 2:r0  3:K 4:symbol
        self.bond_coeffs = np.empty((0, 5), dtype=object)
        # 0:type 1:func 2:theta0 3:K 4:symbol
        self.angle_coeffs = np.empty((0, 5), dtype=object)
        # 0:type 1:func 2:V1 3:V2 4:V3 5:V4 6:symbol
        self.dihedral_coeffs = np.empty((0, 7), dtype=object)
        # 0:type 1:func 2:V1 3:V2 4:V3 5:V4 6:symbol
        self.improper_coeffs = np.empty((0, 7), dtype=object)

    def SetMolName(self):
        """set self.atoms[:, 8:10]
        """
        self.atoms[:, 8] = self.atoms[:, 7]  # ff_symbol
        self.atoms[:, 9] = self.atoms[:, 7]
        _, idx = np.unique(self.atoms[:, 1], return_index=True)
        for i in self.atoms[idx, 1]:
            flag = self.atoms[:, 1] == i
            elems = self.atoms[flag, 7]
            molname = "-".join(elems)
            if molname == "C-O-O-O":
                self.atoms[flag, 9] = "CO3"
            if molname == "O-H-H":
                self.atoms[flag, 9] = "H2O"
                self.atoms[flag, 8] = ["Ow", "Hw", "Hw"]

    def SetAtomCoeff(self):
        type_u, idx = np.unique(self.atoms[:, 8], return_index=True)
        idx = sorted(idx)
        type_u = self.atoms[idx, 8]
        nm = (type_u.shape[0], self.atom_coeffs.shape[1])
        self.atom_coeffs = np.empty(nm, dtype=object)
        for i, t in enumerate(type_u):
            idx = self.atoms[:, 8] == t
            molname = self.atoms[idx, 9][0]
            self.atoms[idx, 2] = i + 1
            self.atoms[idx, 3] = VDW[t][1]
            self.atom_coeffs[i, 0] = i + 1
            self.atom_coeffs[i, 1] = t  # symol
            self.atom_coeffs[i, 2] = molname
            self.atom_coeffs[i, 3] = VDW[t][0]  # mass
            self.atom_coeffs[i, 4] = VDW[t][1]  # charge
            self.atom_coeffs[i, 6:8] = 0.0     # eps sigma

    def SetBondCoeff(self):
        type_u, idx = np.unique(self.bonds[:, 4], return_index=True)
        idx = sorted(idx)
        type_u = self.bonds[idx, 4]
        nm = (type_u.shape[0], self.bond_coeffs.shape[1])
        self.bond_coeffs = np.empty(nm, dtype=object)
        for i, t in enumerate(type_u):
            self.bonds[self.bonds[:, 4] == t, 1] = i + 1
            self.bond_coeffs[i, 0] = i + 1
            self.bond_coeffs[i, 2] = 0.0
            self.bond_coeffs[i, 3] = 0.0
            self.bond_coeffs[i, 4] = t

    def SetAngleCoeff(self):
        type_u, idx = np.unique(self.angles[:, 5], return_index=True)
        idx = sorted(idx)
        type_u = self.angles[idx, 5]
        nm = (type_u.shape[0], self.angle_coeffs.shape[1])
        self.angle_coeffs = np.empty(nm, dtype=object)
        for i, t in enumerate(type_u):
            self.angles[self.angles[:, 5] == t, 1] = i + 1
            self.angle_coeffs[i, 0] = i + 1
            self.angle_coeffs[i, 2] = 0.0
            self.angle_coeffs[i, 3] = 0.0
            self.angle_coeffs[i, 4] = t

    def SetDihedralCoeff(self):
        type_u, idx = np.unique(self.dihedrals[:, 6], return_index=True)
        idx = sorted(idx)
        type_u = self.dihedrals[idx, 6]
        nm = (type_u.shape[0], self.dihedral_coeffs.shape[1])
        self.dihedral_coeffs = np.empty(nm, dtype=object)
        for i, t in enumerate(type_u):
            self.dihedrals[self.dihedrals[:, 6] == t, 1] = i + 1
            self.dihedral_coeffs[i, 0] = i + 1
            self.dihedral_coeffs[i, 2:6] = 0.0
            self.dihedral_coeffs[i, 6] = t

    def SetImproperCoeff(self):
        """set self.impropers[:, 2:6]
        """
        type_u, idx = np.unique(self.impropers[:, 6], return_index=True)
        idx = sorted(idx)
        type_u = self.impropers[idx, 6]
        nm = (type_u.shape[0], self.improper_coeffs.shape[1])
        self.improper_coeffs = np.empty(nm, dtype=object)
        for i, t in enumerate(type_u):
            self.impropers[self.impropers[:, 6] == t, 1] = i + 1
            self.improper_coeffs[i, 0] = i + 1
            self.improper_coeffs[i, 2:6] = 0.0
            self.improper_coeffs[i, 6] = t

    def SetCoeffs(self):
        self.SetAtomCoeff()
        self.SetBondCoeff()
        self.SetAngleCoeff()
        self.SetDihedralCoeff()
        self.SetImproperCoeff()

    def SetImpropers(self):
        """set self.impropers[:, 2:6]
        """
        angles = self.angles[:, 2:5].astype(int)
        ang, ar, c = np.unique(
            angles[:, 1], return_counts=True, return_index=True)
        ang = angles[ar[c == 3]]  # three counts atom(carbon)
        impropers = self.AddBondsLists(ang, improper=True)
        if impropers.shape[0] == 0:
            return
        nm = (impropers.shape[0], self.impropers.shape[1])
        self.impropers = np.empty(nm, dtype=object)
        self.impropers[:, 2:6] = np.sort(impropers, axis=1)
        self.impropers[:, 0] = np.arange(1, self.impropers.shape[0] + 1)

    def SetDihedrals(self):
        """set self.dihedrals[:, 2:6]
        """
        dihedrals = self.AddBondsLists(self.angles[:, 2:5])
        if dihedrals.shape[0] == 0:
            return
        nm = (dihedrals.shape[0], self.dihedrals.shape[1])
        self.dihedrals = np.empty(nm, dtype=object)
        self.dihedrals[:, 2:6] = dihedrals
        flag = self.atoms[dihedrals[:, 3] - 1, 7] != "S"
        self.dihedrals[flag, :] = self.dihedrals[flag, ::-1]
        # self.dihedrals[flag, :] = self.dihedrals[flag, ::-1]

    def SetAngles(self):
        """set self.angles[:, 2:5]
        """
        angles = self.AddBondsLists(self.bonds[:, 2:4])
        # angles = [atom_a, atom_b, atom_c]
        if angles.shape[0] == 0:
            return
        nm = angles.shape[0], self.angles.shape[1]
        self.angles = np.empty(nm, dtype=object)
        self.angles[:, 2:5] = angles

    def SetBonds(self):
        """atom peir in CUTOFF to set self.bonds
        """
        bonds = np.empty((0, 2), dtype=int)
        for k, v in CUTOFF.items():  # k:elm_a-elm_b, v:cutoff)
            elem_a, elem_b = k.split("-")
            idx_a = np.where(self.atoms[:, 7] == elem_a)[0]
            idx_b = np.where(self.atoms[:, 7] == elem_b)[0]
            xyz_j = self.atoms[idx_b, 4:7].astype(float)
            for idx in idx_a:
                xyz_i = self.atoms[idx, 4:7].astype(float)
                dr = xyz_i - xyz_j
                dr = dr - np.rint(dr)  # boundary condition
                dr = np.dot(dr, self.lattice.M)  # scaler product
                r = np.sqrt(np.sum(dr**2, axis=1))
                p = np.where(r < v)[0]  # search bound atom
                bond = np.array([np.repeat(idx, len(p)), idx_b[p]])
                bonds = np.vstack((bonds, bond.T))
                # bonds = [[elem_a, elem_b], ,,,]

        nm = (bonds.shape[0], self.bonds.shape[1])  # (numbnd, 5)
        self.bonds = np.empty(nm, dtype=object)
        self.bonds[:, 2:4] = bonds + 1   # id_a, id_b
        symbol_a = self.atoms[bonds[:, 0], 7]
        symbol_b = self.atoms[bonds[:, 1], 7]
        self.bonds[:, 4] = symbol_a + "-" + symbol_b

    def CheckBonds(self, chain, b, tail=False):
        """hstack other bond atom of chain to chain
        """
        # b:chain[0]
        add_blist = np.empty(0, dtype=int)
        id_a = self.bonds[self.bonds[:, 2] == b][:, 3]
        id_b = self.bonds[self.bonds[:, 3] == b][:, 2]
        ids = np.append(id_a, id_b)
        ids = np.unique(ids)  # either
        for i in ids:  # i:ppbond_atom
            if np.sum(chain == i) == 0:
                add_blist = np.append(add_blist, i)
        add_blist = add_blist.reshape((-1, 1))
        blist = np.repeat([chain], add_blist.shape[0], axis=0)
        if tail is False:  # [[iregular, chain[0], chain[1]],,]
            blist = np.hstack((add_blist, blist))
        else:
            blist = np.hstack((blist, add_blist))
        return blist

    def AddBondsLists(self, chains, improper=False):
        blists = np.empty((0, chains.shape[1] + 1))
        for chain in chains:  # chain:elem_a, elem_b
            if improper is False:
                blist = self.CheckBonds(chain, chain[0], tail=False)
                blists = np.vstack((blists, blist))
                blist = self.CheckBonds(chain, chain[-1], tail=True)
                blists = np.vstack((blists, blist))
            else:
                blist = self.CheckBonds(chain, chain[1], tail=True)
                blists = np.vstack((blists, blist))
        blists = blists.astype(int)  # blist.shape = (N, 3)
        if blists.shape[0] == 0:  # one bond
            return blists
        if improper is True:
            blists = blists[:, [0, 2, 1, 3]]
        else:
            idx = blists[:, 0] > blists[:, -1]
            blists[idx, :] = blists[idx, ::-1]  # inverse
            blists = np.unique(blists, axis=0)
        return blists

    def ShowMolecularInfo(self):
        molname = self.molname
        atoms = self.atoms.shape[0]
        bonds = self.bonds.shape[0]
        angles = self.angles.shape[0]
        dihedrals = self.dihedrals.shape[0]
        impropers = self.impropers.shape[0]
        print("{}: {} atoms".format(molname, atoms))
        print("{}: {} bonds".format(molname, bonds))
        print("{}: {} angles".format(molname, angles))
        print("{}: {} dihedrals".format(molname, dihedrals))
        print("{}: {} impropers".format(molname, impropers))

    def OutputXYZ(self, outfile):
        o = open(outfile, "w")
        atoms = np.copy(self.atoms)
        # atoms[:, 4:7] += self.nxnynz
        atoms[:, 4:7] = np.dot(atoms[:, 4:7], self.lattice.M)
        o.write("{}\n\n".format(self.atoms.shape[0]))
        fmt = "{:2s} {:12.6f} {:12.6f} {:12.6f} # {:4d} {}\n"
        for a in atoms:
            # print(a[7], a[4], a[5], a[6], a[0], a[8])
            o.write(fmt.format(a[7], a[4], a[5], a[6], a[0], a[8]))
        print(outfile, "was created.")

    def LoadCifFile(self, ciffile, super_cell):
        body = open(ciffile).read()
        body = re.sub('\(', '', body)  # deleat '\('
        body = re.sub('\)', '', body)  # deleat '\)'
        # print(body)
        a = float(re.search("_cell_length_a +([0-9\.]+)", body).group(1))
        b = float(re.search("_cell_length_b +([0-9\.]+)", body).group(1))
        c = float(re.search("_cell_length_c +([0-9\.]+)", body).group(1))
        alpha = float(
            re.search("_cell_angle_alpha +([0-9\.]+)", body).group(1))
        beta = float(re.search("_cell_angle_beta +([0-9\.]+)", body).group(1))
        gamma = float(
            re.search("_cell_angle_gamma +([0-9\.]+)", body).group(1))
        self.lattice.a = float(a)
        self.lattice.b = float(b)
        self.lattice.c = float(c)
        self.lattice.alpha = float(alpha)
        self.lattice.beta = float(beta)
        self.lattice.gamma = float(gamma)
        self.lattice.SetMatrix()
        body = body.split("loop_")
        for b in body:
            if re.match("\s+_atom_site_label", b) != None:
                break
        body = b.strip()
        labels = re.findall("_(\S+)", body)  # key name
        lines = body.split("\n")[len(labels):]  # atom data
        data = np.array(" ".join(lines).split(), dtype=object)
        data = data.reshape((-1, len(labels)))
        l_idx = labels.index("atom_site_label")  # atom label
        x_idx = labels.index("atom_site_fract_x")
        y_idx = labels.index("atom_site_fract_y")
        z_idx = labels.index("atom_site_fract_z")
        try:
            e_idx = labels.index("atom_site_type_symbol")  # atom symbol        
            idx = [x_idx, y_idx, z_idx, e_idx, l_idx]
        except ValueError:
            idx = [x_idx, y_idx, z_idx, l_idx, l_idx]
        data = np.array(data[:, idx], dtype=object)

        order = np.empty(0, dtype=int)  # atom_symbol idx
        # All element was not assigned
        elem_order = " ".join(VDW.keys())
        elem_order = elem_order.replace("Ow", "").replace("Hw", "H")
        elem_order = elem_order.split()  # ELEMENT_ORDER
        for e in ELEMENT_ORDER:
            order = np.append(order, np.where(data[:, 3] == e)[0])
        if data.shape[0] != order.shape[0]:
            print(data.shape[0], order.shape[0])
            print("All element was not assigned!!!")
            print("exit")
            exit()
        data = data[order, :]
        nm = (data.shape[0], self.atoms.shape[1])  # (n_atom, 10)
        self.atoms = np.empty(nm, dtype=object)
        self.atoms[:, 4:9] = data
        self.atoms[:, 9] = self.atoms[:, 8]  # atom label
        self.atoms[:, 4:7] = self.atoms[:, 4:7].astype(float)  # xyz
        self.atoms[:, 4:7] -= np.floor(self.atoms[:, 4:7].astype(float))
        self.SetSuperCell(super_cell)
        # self.OutputXYZ('hge.xyz')
        self.atoms[:, 0] = np.arange(1, self.atoms.shape[0] + 1)  # id
        self.SetBonds()       # set self.bonds
        self.SetChains()      # set self.molecules, atoms[0:aid, 1:mid]

        # order = np.lexsort((self.atoms[:, 0], self.atoms[:, 1]))
        order = np.empty(0).astype(int)
        for mol_id in np.unique(self.atoms[:, 1]):
            atom_idxs = np.where(self.atoms[:, 1] == mol_id)[0]
            atoms = self.atoms[atom_idxs, 0].astype(int)
            atom_idxs = atom_idxs[np.argsort(atoms)]
            order = np.append(order, atom_idxs)

        # a_id argsort by m_id(,,C, O, O, O, C, O,,,,)
        self.atoms = self.atoms[order, :]
        self.atoms[:, 0] = np.arange(1, self.atoms.shape[0] + 1)
        self.SetBonds()  # update self.bonds[:, 4]:sym_a-sym_b
        self.SetAngles()  # set self.angles[:, 2:5]
        self.SetDihedrals()  # set self.dihedrals[:, 2:6]
        self.SetImpropers()  # set self.improper[:, 2:6]
        self.SetMolName()  # set self.atoms[:, 8:9]
        lists = [self.bonds, self.angles, self.dihedrals, self.impropers]
        for i, a in enumerate(lists):
            if a.shape[0] == 0:
                continue
            # print(lists[i][:, 2:-1])
            lists[i][:, 0] = np.arange(1, a.shape[0] + 1)
            idxs = (lists[i][:, 2:-1] - 1).astype(int)
            ff = self.atoms[idxs, 8]  # bond_atom
            for j, ff_i in enumerate(ff):
                lists[i][j, -1] = "-".join(ff_i)

    def SetSuperCell(self, super_cell):
        super_num = super_cell[0] * super_cell[1] * super_cell[2]
        super_atoms = np.empty((self.atoms.shape[0]*super_num,
                                self.atoms.shape[1]), dtype=object)
        super_xyz = self.atoms[:, 4:7].reshape(1, -1, 3)
        lx = np.arange(super_cell[0])
        ly = np.arange(super_cell[1])
        lz = np.arange(super_cell[2])
        mx, my, mz = [mesh.reshape(-1, 1) for mesh in np.meshgrid(lx, ly, lz)]
        mesh = np.hstack((mx, my, mz)).reshape(-1, 1, 3)
        super_xyz = ((super_xyz + mesh) / super_cell).reshape(-1, 3)
        
        super_atoms[:, 4:7] = super_xyz
        super_atoms[:, 7:] = np.tile(self.atoms[:, 7:], (super_num, 1))
        self.atoms = super_atoms

        self.lattice.a *= super_cell[0]
        self.lattice.b *= super_cell[1]
        self.lattice.c *= super_cell[2]
        self.lattice.SetMatrix()

        order = np.empty(0, dtype=int)  # atom_symbol idx
        for e in ELEMENT_ORDER:
            order = np.append(order, np.where(self.atoms[:, -3] == e)[0])
        self.atoms = self.atoms[order, :]

                
    def Nchain(self, bonds, chain, chains):
        # bonds:(False;non-check, True;check) :flag 1:id_a 1:id_b
        # chain:[id(False)O, H], chains:[]
        idx_a = np.where((bonds[:, 0] == False) &
                         (bonds[:, 1] == chain[-1]))[0]
        idx_b = np.where((bonds[:, 0] == False) &
                         (bonds[:, 2] == chain[-1]))[0]
        ids_a = bonds[idx_a, 2]
        ids_b = bonds[idx_b, 1]
        ids = np.append(ids_a, ids_b)  # atom(chain_bond)_id
        idx = np.append(idx_a, idx_b)  # atom(chain)_id
        check_ids = np.vstack((ids, idx)).T  # ids, bondのidx
        for c in check_ids:  # c:[ids, idx]
            bonds[c[1], 0] = True
            chain = np.append(chain, c[0])
            chains.append(chain)
            if chain.shape[0] == MAXCHAIN:  # 100
                chain = chain[:-1]
                continue
            self.Nchain(bonds, chain, chains)
            chain = chain[:-1]

    def SetChains(self):
        # flag, id1, id2
        bonds = np.empty((self.bonds.shape[0], 3), dtype=object)
        bonds[:, 1:3] = self.bonds[:, 2:4]  # id_a, id_b
        bonds[:, 0] = False
        atoms = np.empty((self.atoms.shape[0], 2), dtype=object)
        atoms[:, 1] = self.atoms[:, 0]  # id
        atoms[:, 0] = False
        mols = []
        # no-bond
        for i, a in enumerate(atoms):
            v = np.sum(a[1] == bonds[:, 1:3])
            if v == 0:  # a not in CUTOFF
                mols.append(np.array([a[1]]))
                atoms[i, 0] = True
        for a in atoms:
            chains = []
            if a[0] == True:  # check済み, no-bond(=Ca)
                continue
            self.Nchain(bonds, np.array([a[1]]), chains)
            zmats = np.empty((0, 2), dtype=int)
            for chain in chains:  # chain:[a[1], a[1]_bond_id]
                if len(chain) == 2:
                    zmats = np.vstack((zmats, chain))
                if len(chain) > 2:
                    zmat = np.repeat(chain, 2)[1:-1].reshape((-1, 2))
                    zmats = np.vstack((zmats, zmat))  # [[c1, c1],,,]
            zmats_u, idx = np.unique(zmats, return_index=True, axis=0)
            zmats = zmats_u[np.argsort(idx)]  # sort along axis=0
            ids_u = np.unique(zmats)  # all elem of molecule in CUTOFF
            atoms[ids_u - 1, 0] = True
            mols.append(zmats)
        # print(mols)
        # exit()
        self.molecules = mols  # [[elem_1, elem_2], ,,,]
        for i, mol in enumerate(self.molecules):
            idx = np.unique(mol) - 1
            self.atoms[idx, 1] = i + 1  # same id in molecule
        # atoms_idx = np.arange(self.atoms.shape[0], dtype=int)
        # bonds_idx = self.bonds[:, 2:4].astype(int)
        # no_bond_idx = atoms_idx[~np.isin(atoms_idx, bonds_idx)]  # Ca

        # mol_idx = no_bond_idx.size
        # self.atoms[no_bond_idx, 1] = np.arange(mol_idx, dtype=int)        
        # for bond_idx in bonds_idx:
        #     if np.all(self.atoms[bond_idx-1, 1] == None):
        #         self.atoms[bond_idx-1, 1] = mol_idx
        #         mol_idx += 1
        #     elif self.atoms[bond_idx[0]-1, 1] == None:
        #         self.atoms[bond_idx[0]-1, 1] = self.atoms[bond_idx[1], 1]
        #     elif self.atoms[bond_idx[1]-1, 1] == None:
        #         self.atoms[bond_idx[1]-1, 1] = self.atoms[bond_idx[0], 1]
           
        # print(self.atoms[:, 1])
        # exit()


            
    def SetLammpsData(self):
        """make datafile content(head, body)
        """
        head = "molecular system: {}\n".format(self.molname)
        body = ""
        head += "{} atoms\n".format(self.atoms.shape[0])
        head += "{} bonds\n".format(self.bonds.shape[0])
        head += "{} angles\n".format(self.angles.shape[0])
        head += "{} dihedrals\n".format(self.dihedrals.shape[0])
        head += "{} impropers\n".format(self.impropers.shape[0])
        head += "{} atom types\n".format(self.atom_coeffs.shape[0])
        head += "{} bond types\n".format(self.bond_coeffs.shape[0])
        head += "{} angle types\n".format(self.angle_coeffs.shape[0])
        head += "{} dihedral types\n".format(self.dihedral_coeffs.shape[0])
        head += "{} improper types\n".format(self.improper_coeffs.shape[0])
        head += "\n"
        L = self.lattice.lammps_lattice[:, 0:2]
        head += "{:>10.6f} {:>10.6f} xlo xhi\n".format(*L[0])
        head += "{:>10.6f} {:>10.6f} ylo yhi\n".format(*L[1])
        head += "{:>10.6f} {:>10.6f} zlo zhi\n".format(*L[2])
        if np.sum(self.lattice.lammps_lattice[:, 2] == 0.0) != 3:
            L = self.lattice.lammps_lattice[:, 2]
            head += "{:>10.6f} {:>10.6f} {:>10.6f} xy xz yz\n".format(*L)
        body += "\nMasses\n\n"
        fmt = "{:6d} {:10.6f} # {:4s} {:4s}\n"
        for m in self.atom_coeffs:
            body += fmt.format(m[0], m[3], m[1], m[2])
        body += "\nAtoms\n\n"
        fmt1 = "{:6d} {:3d} {:4d} {:10.6f} "
        fmt2 = "{:10.6f} {:10.6f} {:10.6f} "
        fmt3 = "{:2d} {:2d} {:2d} "
        fmt4 = " # {:s} {:s}\n"
        xyz = np.dot(self.atoms[:, 4:7], self.lattice.M)
        for i, r in enumerate(xyz):
            body += fmt1.format(*self.atoms[i, 0:4])
            body += fmt2.format(*r)
            body += fmt3.format(*self.nxnynz[i, :])
            body += fmt4.format(*self.atoms[i, 8:10])
        self.head = head
        self.body = body
        self.SetLammpsBondCoeffs()
        self.SetLammpsAngleCoeffs()
        self.SetLammpsDihedralCoeffs()
        self.SetLammpsImproperCoeffs()

    def SetLammpsBondCoeffs(self):
        # Bonds
        if self.bonds.shape[0] != 0:
            self.body += "\nBonds\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.bonds:
                self.body += fmt.format(b[0], b[1], b[2], b[3], b[4])

    def SetLammpsAngleCoeffs(self):
        # Angles
        if self.angles.shape[0] != 0:
            self.body += "\nAngles\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.angles:
                self.body += fmt.format(b[0], b[1], b[2], b[3], b[4], b[5])

    def SetLammpsDihedralCoeffs(self):
        # Dihedrals
        if self.dihedrals.shape[0] != 0:
            self.body += "\nDihedrals\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            for b in self.dihedrals:
                self.body += fmt.format(b[0], b[1],
                                        b[2], b[3], b[4], b[5], b[6])

    def SetLammpsImproperCoeffs(self):
        # Impropers
        if self.impropers.shape[0] != 0:
            self.body += "\nImpropers\n\n"
            fmt = "{:6d} {:6d} {:6d} {:6d} {:6d} {:6d} # {:s}\n"
            # I,J,K,L
            for b in self.impropers:
                self.body += fmt.format(b[0], b[1],
                                        b[2], b[3], b[4], b[5], b[6])

    def SetImageFlag(self):
        """set self.nxnynz
        """
        def checkbond(i):
            ids_a = self.bonds[self.bonds[:, 2] == i + 1, 3] - 1
            ids_b = self.bonds[self.bonds[:, 3] == i + 1, 2] - 1
            ids = np.append(ids_a, ids_b)  # either
            for j in ids:
                a = self.atoms[i, 4:7] + self.nxnynz[i]
                b = self.atoms[j, 4:7] + self.nxnynz[j]
                n = np.rint((a - b).astype(float)).astype(int)
                if np.any(n != 0):
                    self.nxnynz[j] += n
                    checkbond(j)  # boundary
        self.nxnynz = np.zeros((self.atoms.shape[0], 3), dtype=int)
        for atom in self.atoms:
            checkbond(atom[0] - 1)

    def MakeCfg(self):
        types = {}  # element:id
        types["Ca"] = 1
        types["C"] = 2
        types["O"] = 3
        types["Hw"] = 4
        types["Ow"] = 5
        nums = {}
        idxs = np.empty(0, dtype=int)
        for e in types.keys():
            target_idx = np.where(self.atoms[:, 8] == e)[0]
            idxs = np.append(idxs, target_idx)
            nums[e] = target_idx.shape[0]
        self.cfg = self.atoms[idxs, :]
        xyz = self.cfg[:, 4:7].astype(float)
        xyz = xyz - np.floor(xyz)
        xyz = xyz * 2 - 1
        self.comment = "{} ".format(self.molname)
        for k, v in types.items():
            self.comment += " {}:{}".format(v, k)
        s = "{}\n".format(self.comment)
        s += "ACC\n\n\n"
        s += "0 0 2 moves generated, tried, accepted\n"
        s += "0 configurations saved\n\n"
        s += "{} molecules of all types\n".format(self.atoms.shape[0])
        s += "{} types of molecules\n".format(len(types))
        s += "1 is the largest number of atoms in a molecule\n"
        s += "0 Euler angles are provided\n\n"
        s += "F (box is cubic)\n"
        s += "Defining vectors are:\n"
        s += "{:9.6f}   0.000000   0.000000\n".format(self.lattice.a*0.5)
        s += " 0.000000  {:9.6f}   0.000000\n".format(self.lattice.b*0.5)
        s += " 0.000000   0.000000  {:9.6f}\n\n".format(self.lattice.c*0.5)
        for k, v in nums.items():
            s += "{} molecules of type  {}\n".format(v, types[k])
            s += "1 atomic sites\n"
            s += "0.000000   0.000000   0.000000\n\n"
        for i, r in enumerate(xyz):
            s += "{:20.16f} {:20.16f} {:20.16f}\n".format(*r)
        self.cfgbody = s
        elems = list(types.keys())
        radius = {}
        radius["Ca"] = 0.8
        radius["C"] = 0.5
        radius["O"] = 0.5
        radius["Ow"] = 0.5
        radius["Hw"] = 0.4
        for i, m in enumerate(elems):
            for j, n in enumerate(elems[i:]):
                r = "{:.1f}".format(radius[m]+radius[n])
                print("{:>2s}-{:<2s} {}-{} {}".format(m, n, i+1, j+1, r))

    def MakeFnc(self, minmax):
        self.cfg[:, 0] = np.arange(1, self.cfg.shape[0]+1, dtype=int)
        bonds = {}
        bonds["C-O"] = 1.3042
        bonds["O-C"] = bonds["C-O"]
        bonds["O-O"] = 2 * 1.3042 * np.sin(0.5*120/180*np.pi)
        bonds["Ow-Hw"] = 1.012
        bonds["Hw-Ow"] = bonds["Ow-Hw"]
        bonds["Hw-Hw"] = 2 * 1.012 * np.sin(0.5*113.24/180*np.pi)
        bmax, bmin = [], []
        for b in bonds.values():  # set min max
            bmax += ["{:.8f}".format(b*(1+minmax))]
            bmin += ["{:.8f}".format(b*(1-minmax))]
        bondtypes = {}
        for i, k in enumerate(bonds.keys()):
            bondtypes[k] = i + 1
        body = "{}\n\n".format(self.comment)
        body += "No. of possible rmin-rmax pairs:\n"
        body += "{}\n".format(len(bonds))
        body += "{}\n".format(" ".join(bmin))
        body += "{}\n".format(" ".join(bmax))
        body += "\n{}\n\n".format(self.cfg.shape[0])
        for i, c in enumerate(self.cfg):
            flag = self.cfg[:, 1] == c[1]
            flag[i] = False
            blist = c[8] + "-" + self.cfg[flag, 8]
            idlist = self.cfg[flag, 0]
            body += "{} {}\n".format(c[0], len(blist))  # 1st line
            s, t = [], []
            for i, b in enumerate(blist):
                s += ["{}".format(idlist[i])]  # 2nd line
                t += ["{}".format(bondtypes[b])]  # 3rd line
            if len(s) != 0:
                body += "{}\n".format(" ".join(s))
                body += "{}\n".format(" ".join(t))
        self.fncbody = body


if __name__ == '__main__':
    msg = "make lammps *.data from P1 ciffile"
    par = argparse.ArgumentParser(description=msg)
    par.add_argument('ciffile')
    par.add_argument('-t', '--outtype', default="lammpsdata",
                     choices=["lammpsdata", "cfg"])
    par.add_argument('-s', '--super_cell', nargs=3, type=int, default=[1, 1, 1])
    par.add_argument('--minmax', default=0.01, type=float)
    args = par.parse_args()


    m = MOLECULE(args.ciffile)
    m.LoadCifFile(args.ciffile, args.super_cell)
    basename = args.ciffile[:-4]  # deleat ".cif"
    basename = basename.replace("_P1", "")  # deleat "_P1"
    if args.outtype == "cfg":
        m.MakeCfg()
        m.MakeFnc(args.minmax)
        outfile = basename + ".cfg"
        o = open(outfile, "w")
        o.write(m.cfgbody)
        print(outfile, "was created.")
        fncfile = basename + ".fnc"
        o = open(fncfile, "w")
        o.write(m.fncbody)
        print(fncfile, "was created.")

    if args.outtype == "lammpsdata":
        m.SetImageFlag()
        m.SetCoeffs()
        m.SetLammpsData()
        outfile = basename + ".data"
        o = open(outfile, "w")
        o.write(m.head)
        o.write(m.body + "\n")
        print(outfile, "was created.")
