#!/usr/bin/env python
# import numpy as np
import subprocess as sp
import argparse
import os

par = argparse.ArgumentParser(description="test")
par.add_argument('lammpstrj', help="lammpstrj")
# par.add_argument('-p', '--psf', default=None, help="psf")
par.add_argument('-s', '--size', nargs=2, default=[500, 800])
par.add_argument('-i', '--init', type=int, default=0)
par.add_argument('-o', '--output', action='store_true', default=False)
par.add_argument('-rx', '--rotate_x', type=float, default=90)
par.add_argument('-ry', '--rotate_y', type=float, default=90)
par.add_argument('-z', '--scale', type=float, default=1)
par.add_argument('-wt', '--water_trans', action='store_true', default=False)
args = par.parse_args()

# read index_file
VIEWS = {}
VIEWS["index"] = {}
VIEWS["index"]["layer_oh"] = 1.2
VIEWS["index"]["H2O"] = 1.2
VIEWS["index"]["OH"] = 1.2
VIEWS["index"]["CO3"] = 1.6

# identify atom_type
VIEWS["type"] = {}
VIEWS["type"]["st"] = [[1], 1.8]  # polyhedra
VIEWS["type"]["st-ob"] = [[1, 4, 5], 1.8]  # polyhedra
VIEWS["type"]["cah"] = [[2], None]  # 2
VIEWS["type"]["Ca"] = [[3], None]  # 2
VIEWS["type"]["Cs"] = [[14], None]  # 2
VIEWS["type"]["Cl"] = [[15], None]  # 2

# colors
COLORS = {}
COLORS["st"] = "ColorID 0"
COLORS["st-ob"] = "ColorID 0"
COLORS["cah"] = "ColorID 3"
COLORS["Ca"] = "ColorID 11"
COLORS["layer_oh"] = "ColorID 12"
COLORS["Cl"] = "ColorID 7"
COLORS["Cs"] = "ColorID 5"
COLORS["H2O"] = "ColorID 21"
COLORS["OH"] = "ColorID 11"
COLORS["CO3"] = "ColorID 1"

# methods
METHODS = {}
METHODS["st"] = ["polyhedra"]
METHODS["st-ob"] = ["bond", "vdw"]
METHODS["cah"] = ["vdw"]
METHODS["Ca"] = ["vdw"]
METHODS["Cl"] = ["vdw"]
METHODS["Cs"] = ["vdw"]
METHODS["layer_oh"] = ["vdw", "bond"]
METHODS["H2O"] = ["bond", "vdw"]
# METHODS["H2O"] = ["Qsurf"]
METHODS["OH"] = ["bond", "vdw"]
METHODS["CO3"] = ["bond", "vdw"]

MATERIALS = {}
if args.water_trans == True:
    MATERIALS["H2O"] = "Transparent"
    MATERIALS["OH"] = "Transparent"
else:
    MATERIALS["H2O"] = "RTChrome"
    MATERIALS["OH"] = "RTChrome"
MATERIALS["st"] = "RTChrome"
MATERIALS["st-ob"] = "RTChrome"
MATERIALS["layer_oh"] = "RTChrome"
MATERIALS["cah"] = "RTChrome"
MATERIALS["Ca"] = "RTChrome"
MATERIALS["Cs"] = "RTChrome"
MATERIALS["Cl"] = "RTChrome"
MATERIALS["CO3"] = "RTChrome"


VMD = """
display projection   Orthographic
display depthcue     on
axes location off
menu main on
menu graphics on

# 変数baseをtestをセット
set base test
mol new {trj} type lammpstrj waitfor all
mol delrep 0 top

{bonds}

set ASCALE 1.0
# C
color change rgb 106 0.502 0.286 0.161
color Element C  106
set sel [atomselect top "element C"]
$sel set radius [ expr $ASCALE * 0.77 ]

# H
color change rgb 101 1.0 0.8 0.8
color Element H 101
set sel [atomselect top "element H"]
$sel set radius [ expr $ASCALE * 0.46 ]
# O
color change rgb 108 0.9961 0.0118 0.0000
color Element O  108
set sel [atomselect top "element O"]
$sel set radius [ expr $ASCALE * 0.74 ]
# Ca
color change rgb 120 0.3529 0.5882 0.7412
color Element Ca  120
set sel [atomselect top "element Ca"]
$sel set radius [ expr $ASCALE * 1.97 ]
# Si
color change rgb 114 0.1059 0.2314 0.9804
color Element Si  114
set sel [atomselect top "element Si"]
$sel set radius [ expr $ASCALE * 1.18 ]

# updateするための関数
{update}
# https://www.ks.uiuc.edu/Research/vmd/vmd-1.7.1/ug/node140.html
# selectionをhookする。
trace variable vmd_frame($molid) w do_update


# pbc wrap
# pbc box -color blue -width 1.0 -resolution 8 -on
pbc box -off
display projection Orthographic
axes location off
menu main on
menu graphics on

rotate x to {r_x}
rotate y by {r_y}
# rotate z to 90
# rotate y by -15
# rotate x by 15
scale to 0.02
scale by {scale}
color Display Background white
# mol showrep 0 0 on
# mol showrep 0 1 on
# mol showrep 0 2 on
# mol showrep 0 3 off
# mol showrep 0 4 off
display culling on
display depthcue off

# 0ステップに移動
animate goto {step}
# render snapshot {trj}.tga
# render POV3 {trj}.pov
"""


class VMDdata():
    def __init__(self, lammpstrj):
        self.basename, _ = os.path.splitext(lammpstrj)
        self.lammpstrj = lammpstrj
        # self.psf = args.psf
        # if self.psf is None:
        #     self.psf = f"{self.basename}.psf"
        self.set_vmdtemplate()
        self.tcl = f"{self.basename}.tcl"
        self.vmdcommand()

    def set_vmdtemplate(self):
        template = "mol representation {command}\n"
        template += "mol color {{color}}\n"  # color
        template += "mol selection {{select}}\n"
        template += "mol material {{material}}\n"
        template += "mol addrep top\n"

        self.com = {}
        self.com["bond"] = template.format(
            command="DynamicBonds {cutoff} 0.2 9.0")
        self.com["vdw"] = template.format(
            command="VDW 0.5 12")
        self.com["polyhedra"] = template.format(
            command="Polyhedra {cutoff}")
        self.com["Qsurf"] = template.format(
            command="QuickSurf 1.4 0.3 0.5 MAx")

    def set_update(self, update_dict):
        template = "proc do_update {args} {\n"
        template += "    global molid\n"
        modselect = ""
        reference = ""
        open_index = ""
        get_index = ""
        cnt = 0
        for name, select_ids in update_dict.items():
            template += f"    global update_{name}\n"
            reference += f"set update_{name}(0) none\n"
            for select_id in select_ids:
                modselect += f'        mol modselect {select_id} '
                modselect += f'$molid "$update_{name}($frame)"\n'
            if cnt == 0:
                infomation = "    if {[info exists update_" + name
                infomation += "($frame)]} then {\n"
                
            obj = f"{chr(97+cnt)}p"
            open_index += f'set {obj} [open "{name}.index" r]\n'
            get_index += f"    set update_{name}($i) [gets ${obj}]\n"
            cnt += 1
        template += "    set frame [molinfo $molid get frame]\n"
        template += infomation
        template += modselect
        template += "    }\n}\n"
        template += "# representationのselidを変数にセット\n"
        template += "# 定義したfuncはdo_updateでglobaとして参照する\n"
        template += "set molid [molinfo top]\n"
        template += reference
        template += "# 色を変えるindexファイルを読み込んでセット\n"
        template += open_index
        template += "set n [molinfo $molid get numframes]\n"
        template += "for {set i 0} {$i < $n} {incr i} {\n"
        template += get_index
        template += "}\n"
        return template
        
    def vmdcommand(self):
        bonds_s = ""
        repre_id = 0
        update_dict = {}
        for select_type, repre_dict in VIEWS.items():
            for name, repre_list in repre_dict.items():
                if select_type == "type":
                    types = repre_list[0]
                    cutoff = repre_list[1]
                    select_num = " ".join([f"{i}" for i in types])
                    select = "{" + select_type + " " + select_num + "}"
                else:  # index
                    cutoff = repre_list
                    select = "{none}"
                    update_dict[name] = []
                material = MATERIALS[name]
                bonds_s += f"# {name}\n"
                color = COLORS[name]
                for method in METHODS[name]:
                    template = self.com[method]
                    bonds_s += template.format(cutoff=cutoff,
                                               color=color,
                                               select=select,
                                               material=material)
                    if select_type == "index":
                        key = f"{name}"

                        update_dict[key].append(repre_id)
                    repre_id += 1
                bonds_s += "\n"
        update = self.set_update(update_dict)
        content = VMD.format(trj=self.lammpstrj, bonds=bonds_s, update=update,
                             step=args.init, scale=args.scale,
                             r_x=args.rotate_x, r_y=args.rotate_y)
        if args.output == True:
            content += f"render snapshot {self.basename}.tga\n"
            content += "exit"
        with open(f"{self.tcl}", "w") as o:
            o.write(content)

    def Run(self):
        uname = sp.run("uname", capture_output=True).stdout.decode().strip()
        if uname == "Darwin":  # MacOSXq
            vmd = "/Applications/VMD\ 1.9.4a57-arm64-Rev12.app/"
            # vmd += "Contents/Resources/VMD.app/Contents/MacOS/VMD"
            vmd += "Contents/vmd/vmd_MACOSXARM64"
        if uname == "Linux":
            vmd = "~/local/bin/vmd"  # edit HERE!
        com = f"{vmd} -e {self.tcl} -startup /dev/null"
        com += " -size {} {}".format(*args.size)
        sp.run(com, shell=True)
        # sp.run(f"rm {self.vmdrc}", shell=True)


vmd = VMDdata(args.lammpstrj)
vmd.Run()
