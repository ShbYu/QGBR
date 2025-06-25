from __future__ import print_function, division
import numpy as np
import itertools
from ase.data import covalent_radii
import networkx as nx
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import MinimumDistanceNN, MinimumVIRENN, VoronoiNN


class GraphGenerator:
    def __init__(self):
        pass

    def build_graph(self, atoms, coords, dim=None, maxtry=50, randGen=False, is2D=False):
        """
        Build QG from atoms.
        atoms: (ASE.Atoms) the input crystal structure.
        coords: a list containing coordination numbers for all atoms.
        dim: int or None, the target dimension. If None, do not restrict dimension.
        maxtry: max try times
        randGen: If True, randomly generate quotient graph when direct generation fails
        is2D: If False, use 3x3x3 supercell. If True, use 3x3x1 supercell.
        """
        self.originAtoms = atoms
        Nat = len(atoms)

        disp, pair, D, ratio = get_dispos(atoms, is2D)
        self.disp = disp
        self.pair = pair
        self.D = D
        self.ratio = ratio

        sortInd = np.argsort(ratio)
        ats_num = atoms.get_atomic_numbers()
        ## Search for optimal connectivity
        edges, edgeInds = search_edges(sortInd, coords, pair, ats_num)

        G = nx.MultiGraph()
        for i in range(Nat):
            G.add_node(i)

        if edges is not None:
            for edge, ind in zip(edges, edgeInds):
                m,n,i,j,k = edge
                G.add_edge(m,n, vector=[i,j,k], direction=(m,n), ratio=ratio[ind], bond='connected')

        if nx.number_connected_components(G) == 1:
            self.randG = G
            return G
        else:
            self.randG = None
            return None

def get_dispos(atoms, is2D=False):
    Nat = len(atoms)
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    ## Calculate relative vectors between different atoms in the same cell, and save the indices and cell offset, and sum of covalent radius
    disp = []
    pair = []
    rsum = []
    for m,n in itertools.combinations(range(Nat), 2):
        disp.append(pos[n] - pos[m])
        pair.append([m,n,0,0,0])
        rsum.append(covalent_radii[numbers[m]] + covalent_radii[numbers[n]])
    inDisp = np.array(disp)
    inPair = np.array(pair)
    inRsum = rsum[:]
    ## Calculate relative vectors between atoms in the origin cell and atoms in the surrounding 3x3x3 supercell
    negative_vec = lambda vec: tuple([-1*el for el in vec])
    duplicate = []
    if is2D == True:
        offsetIter = itertools.product(range(-1, 2),range(-1, 2),[0])
    else:
        offsetIter = itertools.product(range(-1, 2),range(-1, 2),range(-1, 2))

    for offset in offsetIter:
        if offset != (0,0,0):
            cellDisp = np.dot(offset, cell)
            if Nat > 1:
                newDisp = inDisp + cellDisp
                newPair = inPair.copy()
                newPair[:,2:5] = offset     # Sign the adjcent cell
                disp.extend(list(newDisp))
                pair.extend(newPair.tolist())
                rsum.extend(inRsum)
            ## Consider equivalent atoms in two different cells
            if negative_vec(offset) not in duplicate:
                duplicate.append(offset)
                for ii in range(Nat):
                    disp.append(cellDisp)
                    pair.append([ii, ii, offset[0], offset[1], offset[2]])
                    rsum.append(2*covalent_radii[numbers[ii]])

    disp = np.array(disp)
    D = np.linalg.norm(disp, axis=1)
    for i in range(len(D)):
        D[i] = int(D[i]*10000) / 10000    
    ratio = D/rsum

    return disp, pair, D, ratio

def search_edges(indices, coords, pairs):
    """
    search for edges so that coordination numbers are fulfilled.
    """
    assert len(indices) == len(pairs)
    edges = []
    edgeInds = []
    #curCoords = np.array([len(row) for row in coords])
    curCoords = np.array(coords)
    for ind in indices:
        m,n,_,_,_ = pairs[ind]
        if (curCoords == 0).all():
            break
        if curCoords[m] > 0 and curCoords[n] > 0:
            edges.append(pairs[ind])
            edgeInds.append(ind)
            curCoords[m] -= 1
            curCoords[n] -= 1 
    #return edges, edgeInds
    if (curCoords == 0).all():
        return edges, edgeInds
    else:
        return None, None

def get_neighbors_of_site_with_index(struct, n, approach, delta, cutoff):

    if approach == "min_dist":
        neighs_list = MinimumDistanceNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    if approach == "min_VIRE":
        neighs_list = MinimumVIRENN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    if approach == "voronoi":
        neighs_list = VoronoiNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)

    return neighs_list

def quot_gen(ats, approach, delta, add_ratio=True):
    '''
    Construct graph using NN finding algorithm.
    ats: atoms
    approach: NN finding algorithm
    delta: tolerance
    '''
    struct = AseAtomsAdaptor.get_structure(ats)
    radius = [covalent_radii[number] for number in ats.get_atomic_numbers()]
    cutoff = ats.cell.cellpar()[:3].mean()
    pos = np.array(ats.get_positions())
    cell = np.array(ats.get_cell())
    G = nx.MultiGraph()
    for i in range(len(struct)):
        G.add_node(i)

    for i in range(len(struct)):
        neighs_list = get_neighbors_of_site_with_index(struct,i,approach,delta,cutoff)
        for nn in neighs_list:
            j = nn['site_index']
            if i <= j:
                rsum = radius[i] + radius[j]
                distance = np.linalg.norm(pos[j] + np.dot(np.array(nn['image']), cell)  - pos[i])
                ratio = distance / rsum
                if add_ratio:
                    G.add_edge(i,j, vector=np.array(nn['image']), direction=(i,j), ratio=ratio)
                else:
                    G.add_edge(i,j, vector=np.array(nn['image']), direction=(i,j))

    if nx.number_connected_components(G) != 1:
        return G
    else:
        return G

def get_dict(atoms):
    '''
    Calculate bond lengths of atom speices
    Return: Dict: {(spe1, spe2): average bond length}
    '''
    dist = {}
    numbers = atoms.get_atomic_numbers()
    ats_spe = np.unique(atoms.get_atomic_numbers())
    for i, j in itertools.combinations_with_replacement(ats_spe, 2):
        dist[tuple(sorted([i, j]))] = [0,0]
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    cutoff = atoms.cell.cellpar()[:3].mean()
    struct = AseAtomsAdaptor.get_structure(atoms)
    for m in range(len(struct)):
        neighs_list = get_neighbors_of_site_with_index(struct,m,approach='min_dist',delta=0.1,cutoff=cutoff)
        for nn in neighs_list:
            n = nn['site_index']
            vector = np.array(nn['image'])
            posn = pos[n] + np.dot(vector, cell)
            disp = np.linalg.norm(np.array(posn - pos[m]), axis=0)
            spem = numbers[m]
            spen = numbers[n]
            dist[tuple(sorted([spem, spen]))][0] += disp
            dist[tuple(sorted([spem, spen]))][1] += 1
            
    for spe in dist.keys():
        if dist[spe][1]:
            dist[spe] = round(dist[spe][0] / dist[spe][1],2)
        else:
            _, pair, D, _ = get_dispos(atoms)
            sortInd = np.argsort(D)
            for ind in sortInd:
                m,n,_,_,_ = pair[ind]
                spem = numbers[m]
                spen = numbers[n]
                if (spem,spen) == spe or (spen,spem) == spe:
                    dist[tuple(sorted([spem, spen]))] = round(D[ind],2)
                    break

    return dist
