## Crystal Quotient Graph
from __future__ import print_function, division
from ase.optimize import BFGS
from ase.constraints import FixAtoms, ExpCellFilter, FixSymmetry
from spglib import get_symmetry_dataset
from ase.io.trajectory import Trajectory
from pycqg.calculator import GraphCalculator
from pycqg.generator import GraphGenerator, quot_gen, get_dict
import multiprocessing as mp
import numpy as np
import argparse
import yaml

def attach_graph(atoms, approach="min_dist", delta=0.1, coor_num=None):
    '''
    attach quotient graph to atoms using NN finding algorithm or given coordination numbers
    coor_num: the given coordination numbers
    '''
    if coor_num is not None:
        gen = GraphGenerator()
        randG = gen.build_graph(atoms, coor_num)
    else:
        randG = quot_gen(atoms, approach, delta)

    if randG is not None:
        atoms.info['graph'] = randG.copy()
        return True
    else:
        return False

def refine(atoms, isif=2, cmax=1, cmin=0.9, cunbond=1, k=2, fmax=1e-5, best_dict=None, steps=200, i=0):
    '''
    refine atoms under the guidance of graph
    isif: refinement option, like vasp, borrowed from Qiuhan Jia
    fmax: convergence criterion of force
    best_dict:
    step: max steps for refinement
    '''
    calc = GraphCalculator(k1=k, k2=k, cmax=cmax, cmin=cmin, cunbond=cunbond, best_dict=best_dict)
    atoms.calc = calc

    fix_atoms_constraint = FixAtoms(mask=[True] * len(atoms))
    if isif == 2:       # lattice fixed
        cell_filter = atoms
    elif isif == 3:     # no fixed
        cell_filter = ExpCellFilter(atoms)
    elif isif == 4:     # volume fixed
        cell_filter = ExpCellFilter(atoms, constant_volume=True)
    elif isif == 5:     # position and volume fixed
        atoms.set_constraint(fix_atoms_constraint)
        cell_filter = ExpCellFilter(atoms, constant_volume=True)
    elif isif == 6:     # position fixed
        atoms.set_constraint(fix_atoms_constraint)
        cell_filter = ExpCellFilter(atoms)
    elif isif == 7:     # only volume varies
        atoms.set_constraint(fix_atoms_constraint)
        cell_filter = ExpCellFilter(atoms, hydrostatic_strain=True)
    elif isif == 8:     # cell shape fixed
        cell_filter = ExpCellFilter(atoms, hydrostatic_strain=True)
    elif isif == 9:     # symmetry fixed
        atoms.set_constraint(FixSymmetry(atoms, 0.1))
        cell_filter = ExpCellFilter(atoms)
    elif isif == 10:     # best wyckoff positions fixed
        data = get_symmetry_dataset(atoms, symprec=0.1)
        wyckoffs = data['wyckoffs']
        fix_wcf_constraint = FixAtoms(mask=[letter==min(wyckoffs) for letter in wyckoffs])
        atoms.set_constraint(fix_wcf_constraint)
        cell_filter = atoms

    class graphOpt(BFGS):
        # using BFGS optimizer, with a little modify
        def converged(self, forces=None):
            if forces is None:
                forces = self.atoms.get_forces()
            return (forces**2).sum(axis=1).max() <= self.fmax**2

    logfile = f"test{i}.txt"
    opt = graphOpt(cell_filter, maxstep=1, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    with open(logfile, 'r+') as fileobj:
        tail = fileobj.readlines()[-1]
        atoms.info['steps'] = int(tail.split()[1])
        fileobj.truncate(0)
    atoms.info['quot_energy'] = atoms.get_potential_energy() / len(atoms)
    atoms.calc = None
    return atoms

def mp_refine(mp_args):
    '''
    mutiprocessing for atoms refinement
    '''
    traj_path, numbers, approach, delta, cmax, cmin, cunbond, k, fmax, steps, energy_cov, best_dict, i = mp_args
    fmax = float(fmax)
    qgbr = Trajectory(f"QGBR_{i}.traj", mode='a')
    traj = Trajectory(traj_path)
    for num in numbers:
        atoms = traj[num] 
        if attach_graph(atoms, approach, delta):
            atoms = refine(atoms, 2, cmax, cmin, cunbond, k, fmax, best_dict, steps, i)
            if atoms.info['quot_energy'] < energy_cov:
                qgbr.write(atoms, append=True)
    qgbr.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_para', type=str, default=None, help='input file', required=True)
    args = parser.parse_args()

    with open(args.input_para, 'r') as fileobj:
        para = yaml.safe_load(fileobj)

    file_path = para["file_path"]
    num_workers = para["num_workers"]
    traj = Trajectory(file_path)[:20]
    total_numbers = np.array(list(range(len(traj))))
    numbers = np.array_split(total_numbers, num_workers)

    if para["use_atoms"]:
        atoms_path = para["atoms_path"]
        bestats = Trajectory(atoms_path)[0]
        ats_dict = get_dict(bestats)
    else:
        ats_dict = None
    
    mp_args = [
        (
            file_path,
            numbers[i],
            para["Graph"]["method"],
            para["Graph"]["delta"],
            para["Refine"]["cmax"],
            para["Refine"]["cmin"],
            para["Refine"]["cunbond"],
            para["Refine"]["k"],
            para["Refine"]["fmax"],
            para["Refine"]["steps"],
            para["Refine"]["energy_cov"],
            ats_dict,
            i,
        )
        for i in range(num_workers)]
    pool = mp.Pool(num_workers)
    pool.imap(mp_refine, mp_args)
    pool.close()
    pool.join()