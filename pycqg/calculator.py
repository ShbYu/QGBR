from __future__ import print_function, division
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
import itertools
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

def add_force(Xmin, Xmax, dvec, D, k):
    # when the distance is too small
    if D > 1e-3:
        uvec = dvec/D
    else:
        dvec = np.random.rand(3)
        uvec = dvec/np.linalg.norm(dvec)    # Can't judge the direction, choose random direction
    if Xmin <= D <= Xmax:
        return 0, 0
    else:
        if D < Xmin:
            energy_at = 0.5*k*(D-Xmin)**2
            f = k*(D-Xmin)*uvec
        elif D > Xmax:
            energy_at = 0.5*k*(D-Xmax)**2
            f = k*(D-Xmax)*uvec
        return energy_at, f

def add_replus(Xmax, dvec, D, k):
    # when the distance is too small
    if D > 1e-3:
        uvec = dvec/D
    else:
        dvec = np.random.rand(3)
        uvec = dvec/np.linalg.norm(dvec)    # Can't judge the direction, choose random direction
    if D >= Xmax:
        return 0, 0
    else:
        energy_at = 0.5*k*(D-Xmax)**2
        f = k*(D-Xmax)*uvec
        return energy_at, f

class GraphCalculator(Calculator):
    """
    Calculator written for optimizing crystal structure based on an initial quotient graph.
    I design the potential so that the following criteria are fulfilled when the potential energy equals zero:
    Suppose there are two atoms A, B. R_{AB} = r_{A} + r_{B} is the sum of covalent radius.
    If A,B are bonded, cmin <= d_{AB}/R_{AB} <= cmax.
    If A,B are not bonded, cunbond <= d_{AB}/R_{AB}.
    """
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {
        'cmin': 0.5,
        'cmax': 1,
        'cunbond': 1.05,
        'k1': 1e-1,
        'k2': 1e-1,
        'useGraphRatio': False,
        'useGraphK': False,
        'ignorePairs': [], # pairs of ignored pairs of atomic numbers
        'best_dict': None,
    }
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        graph = atoms.info['graph'].copy()
        Nat = len(atoms)
        assert Nat == len(graph), "Number of atoms should equal number of nodes in the initial graph!"

        numbers = atoms.get_atomic_numbers()
        unqNumber = set(numbers.tolist()) # unique atomic numbers

        cmax = self.parameters.cmax
        cmin = self.parameters.cmin
        cunbond = self.parameters.cunbond
        k1 = self.parameters.k1
        k2 = self.parameters.k2
        useGraphRatio = self.parameters.useGraphRatio
        useGraphK = self.parameters.useGraphK
        best_dict = self.parameters.best_dict
        ignorePairs = [tuple(sorted(p)) for p in self.parameters.ignorePairs]

        # sums of radius
        if best_dict == None:
            radArr = covalent_radii[atoms.get_atomic_numbers()]
            radArr = np.expand_dims(radArr,1)
            rsumMat = radArr + radArr.T

        # set string constant
        if useGraphK:
            pass
        else:
            for edge in graph.edges(data=True, keys=True):  # Add an attribute k in data
                m,n, key, data = edge
                graph[m][n][key]['k'] = k1

        energy = 0.
        forces = np.zeros((Nat, 3))
        stress = np.zeros((3, 3))

        ## offsets for all atoms
        spos = atoms.get_scaled_positions(wrap=False)
        # remove little difference
        spos[np.abs(spos)<1e-8] = 0
        offsets = np.zeros_like(spos)
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        #print(np.round(pos[27], 4))
        pairs = []
        ## Consider bonded atom pairs
        for edge in graph.edges(data=True, keys=True):
            m,n, key, data = edge
            i,j = data['direction']
            # modify the quotient graph according to offsets
            edgeVec = data['vector']
            n1, n2, n3 = offsets[j] - offsets[i] + data['vector']

            graph[m][n][key]['vector'] = edgeVec
            pairs.append((i,j,n1,n2,n3))
            if best_dict == None:
                rsum = rsumMat[i,j]
            else:
                spem = numbers[m]
                spen = numbers[n]
                rsum = best_dict[tuple(sorted([spem, spen]))]
            # Use ratio attached in graph or not
            if useGraphRatio:
                cmin, cmax = data['ratios']
                assert cmin < cmax, "cmin should be less than cmax!"
            Dmin = cmin * rsum
            Dmax = cmax * rsum
            cells = np.dot(edgeVec, cell)
            dvec = pos[j] + cells - pos[i]
            D = np.linalg.norm(dvec)
            energy_at, f = add_force(Dmin, Dmax, dvec, D, data['k'])
            if energy_at == 0:
                continue
            else:
                energy += energy_at
                forces[i] += f
                forces[j] -= f
                stress += np.dot(f[np.newaxis].T, dvec[np.newaxis])

        pos = atoms.get_positions()
        ## Consider unbonded atom pairs
        cutoffs = dict()
        for num1, num2 in itertools.combinations_with_replacement(unqNumber,2):
            if tuple(sorted([num1,num2])) not in ignorePairs:
                if best_dict == None:
                    cutoffs[(num1,num2)] = cunbond*(covalent_radii[num1]+covalent_radii[num2])
                else:
                    cutoffs[(num1,num2)] = cunbond*best_dict[tuple(sorted([num1,num2]))]

        for i, j, S in zip(*neighbor_list('ijS', atoms, cutoffs)):
            if i <= j:
                # if i,j is bonded, skip this process
                n1,n2,n3 = S
                pair1 = (i,j,n1,n2,n3)
                pair2 = (j,i,-n1,-n2,-n3)
                if pair1 in pairs or pair2 in pairs:
                    continue
                if best_dict == None:
                    rsum = rsumMat[i,j]
                else:
                    spem = numbers[m]
                    spen = numbers[n]
                    rsum = best_dict[tuple(sorted([spem, spen]))]

                Dmax = cunbond * rsum
                cells = np.dot(S, cell)
                dvec = pos[j] + cells - pos[i]
                D = int(10000 * np.linalg.norm(dvec)) / 10000
                if D > 1e-3:
                    uvec = dvec/D
                else:
                    dvec = np.random.rand(3)
                    D = np.linalg.norm(dvec)
                    uvec = dvec/D
                if D < Dmax:
                    energy += 0.5*k2*(D-Dmax)**2
                    f = k2*(D-Dmax)*uvec
                    forces[i] += f
                    forces[j] -= f
                    stress += np.dot(f[np.newaxis].T, dvec[np.newaxis])
        
        stress = 1*stress/atoms.get_volume()
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
