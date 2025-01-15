from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import random
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import random
import pickle
from rdkit.Chem.rdmolops import CombineMols
from rdkit.Chem.rdmolops import SplitMolByPDBResidues
from rdkit.Chem import rdFreeSASA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem import AllChem
import copy
import os
import pandas as pd
from rdkit.Chem.rdmolops import CombineMols
from rdkit.Chem.rdForceFieldHelpers import GetUFFVdWParams
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import rdmolops
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists, CalculateTorsionAngles
import math
from tqdm import tqdm
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.Lipinski import RotatableBondSmarts

def prepare_and_optimize_molecule_from_pdb(pdb_file_path):
    # Step 1: Load the molecule from a PDB file
    mol = Chem.MolFromPDBFile(pdb_file_path, removeHs=False)
    
    if mol is None:
        raise ValueError("Could not read the molecule from PDB file.")
    return mol


def calculate_uff_energy(molecule):
    force_field = AllChem.UFFGetMoleculeForceField(molecule)
    energy = force_field.CalcEnergy()
    return energy


def calculate_internal_vdw_energy(molecule):
    total_energy = 0
    num_atoms = molecule.GetNumAtoms()
    conformer = molecule.GetConformers()[0]
    positions = np.array(conformer.GetPositions())
    distance_matrix = distance_matrix(positions, positions)
    adjacency_matrix = GetAdjacencyMatrix(molecule)
    topological_distance_matrix = GetDistanceMatrix(molecule)
    for i1 in range(num_atoms):
        for i2 in range(0, i1):
            params = GetUFFVdWParams(molecule, i1, i2)
            if params is None:
                continue
            d, e = params
            d = d * 1.0
            if adjacency_matrix[i1, i2] == 1:
                continue
            if topological_distance_matrix[i1, i2] < 4:
                continue
            total_energy += e * ((d / distance_matrix[i1, i2]) ** 12 -
                                 2 * ((d / distance_matrix[i1, i2]) ** 6))
    return total_energy


def get_torsion_energy(molecule):
    molecule_properties = ChemicalForceFields.MMFFGetMoleculeProperties(molecule)
    if molecule_properties is None:
        return 0.0
    force_field_terms = ("Bond", "Angle", "StretchBend", "Torsion", "Oop", "VdW", "Ele")
    term_to_calculate = "Torsion"
    for term in force_field_terms:
        state = (term == term_to_calculate)
        set_method = getattr(molecule_properties, "SetMMFF" + term + "Term")
        set_method(state)
    force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(molecule, molecule_properties)
    energy = force_field.CalcEnergy()
    return energy


def calculate_torsion_energy(molecule):
    energy = 0
    torsion_list, torsion_list_ring = CalculateTorsionLists(molecule)
    angles = CalculateTorsionAngles(molecule, torsion_list, torsion_list_ring)
    for idx, torsion in enumerate(torsion_list):
        indices, _ = torsion
        indices, angle = indices[0], angles[idx][0][0]
        params = rdForceFieldHelpers.GetUFFTorsionParams(molecule, indices[0], indices[1],
                                                         indices[2], indices[3])
        hybridizations = [str(molecule.GetAtomWithIdx(i).GetHybridization()) for i in indices]
        if set([hybridizations[1], hybridizations[2]]) == set(["SP3", "SP3"]):
            n, pi_zero = 3, math.pi
        elif set([hybridizations[1], hybridizations[2]]) == set(["SP2", "SP3"]):
            n, pi_zero = 6, 0.0
        else:
            continue
        energy += 0.5 * params * (1 - math.cos(n * pi_zero) *
                                  math.cos(n * angle / 180 * math.pi))
    return energy


def process_pdb_folder(folder_path, files_not_permitted):
    data = []
    pdb_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdb')]
    
    for filename in tqdm(pdb_files, desc='Processing PDB files'):
        if filename in files_not_permitted:
            print(f"Skipping file {filename} as it is not permitted.")
            continue
        
        pdb_file_path = os.path.join(folder_path, filename)
        mol_name = os.path.splitext(filename)[0]
        mol = prepare_and_optimize_molecule_from_pdb(pdb_file_path)
        uff_energy = calculate_uff_energy(mol)
        internal_vdw_energy = calculate_internal_vdw_energy(mol)
        torsion_energy = get_torsion_energy(mol)
        torsion_angle_energy = calculate_torsion_energy(mol)
        
        data.append({
            'Folder': mol_name,
            'uff_energy': uff_energy,
            'internal_vdw_energy': internal_vdw_energy,
            'torsion_energy': torsion_energy,
            'torsion_angle_energy': torsion_angle_energy
        })

    df = pd.DataFrame(data)
    return df

folder_path = '/path/to/pdb/folder'
files_not_permitted = ['mol_134079.pdb', 'mol_134068.pdb', 'mol_67492.pdb', 'mol_67493.pdb', 'mol_67491.pdb', 
                       'mol_134069.pdb', 'mol_67490.pdb', 'mol_134076.pdb', 'mol_110604.pdb', 'mol_3786.pdb']

df = process_pdb_folder(folder_path, files_not_permitted)



def get_hydrophobic_atom(m):
    n = m.GetNumAtoms()
    retval = np.zeros((n,))
    for i in range(n):
        a = m.GetAtomWithIdx(i)
        s = a.GetSymbol()
        if s.upper() in ["F", "CL", "BR", "I"]:
            retval[i] = 1
        elif s.upper() in ["C"]:
            n_a = [x.GetSymbol() for x in a.GetNeighbors()]
            diff = list(set(n_a) - set(["C"]))
            if len(diff) == 0:
                retval[i] = 1
        else:
            continue
    return retval


def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', value=0.):
    if not maxlen:
        maxlen = max(len(s) for s in sequences)
    
    padded_sequences = np.full((len(sequences), maxlen), fill_value=value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen: 
            padded_sequences[i] = seq[:maxlen]
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            elif padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
    return padded_sequences

def process_hidro(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):
            file_path = os.path.join(folder_path, filename)
            mol = Chem.MolFromPDBFile(file_path, removeHs=False)
            if mol is not None:  # Ensure the molecule could be read
                hydrophobic_atoms = get_hydrophobic_atom(mol)
                data.append((filename, hydrophobic_atoms))
            else:
                print(f"Warning: Failed to read {filename}")
    
    # Extract filenames and sequences
    filenames, sequences = zip(*data)
    
    # Pad the sequences
    padded_sequences = pad_sequences(sequences)
    
    # Construct the DataFrame
    df_data = {'Folder': filenames}
    for i in range(padded_sequences.shape[1]):
        df_data[f'hp_Atom_{i}'] = padded_sequences[:, i]
    df = pd.DataFrame(df_data)
    
    return df
