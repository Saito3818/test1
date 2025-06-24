# test1

* plot
 - logplot.py: lammpsで吐き出されるthermoファイルをplotするコード
 - grplot.py: lammpsで吐き出されるgrファイルをplotするコード
 - hills_plot.py: 時間ごとのHILLSファイル(CVが1つ)をplotするコード
 - hills_2dim_plot.py: HILLSファイル(CVが2つ)をplotするコード
 - plot_pes.py: lammpsで吐き出されるthermoファイルからPESを計算するコード
 - plotzstep.py: lammpsのdumpファイルからz軸分布をplotする
 - numXRD2dim_mean.py:  lammpsのdumpファイルからxrdをplotする
 - cv_time_plot.py:  plumedで吐き出されたCOLVARファイルをplotする

* make_data
 - cif2data.py: CIFファイルからlammps用のデータを作成するコード
 - dump2cif.py: lammpsのdumpファイルをCIFファイルに変換する
 - make_plumeddata.py: lammpsのdataファイルからMetaD用のデータを作成

* other
 - vmd_viewer.py: vmdでdumpファイルを可視化する際に分子系を色分けする
 - symmetry_check.py: spglibでdumpファイルを時間ごとに対称性を評価