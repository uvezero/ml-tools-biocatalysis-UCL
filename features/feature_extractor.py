import numpy as np
import os
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from glob import glob
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import warnings
warnings.filterwarnings('ignore')
import scipy.sparse
from natsort import natsorted
from MDAnalysis.lib.distances import (
           capped_distance,
           self_distance_array, distance_array,  # legacy reasons
)
from MDAnalysis.lib.c_distances import contact_matrix_no_pbc, contact_matrix_pbc
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib.distances import calc_bonds
import logging
logger = logging.getLogger("MDAnalysis.analysis.distances")
import os, subprocess
import csv
from rdkit.Chem import DataStructs,AllChem
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed as e
import natsort
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticHeterocycles
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

new_resid_list = ["resname GTP","resname MG","resid 789", "resid 13", "resid 60", "resid 59", "resid 16", "resid 76","resid 35","resid 19", "resid 18", "resid 785", "resid 15","resid 30","resid 14","resid 117","resid 116","resid 120","resid 119","resid 146","resid 145", "resid 147","resid 786","resid 61","resid 29"]
prot_dir = '/home/juan/Documents/Master_Project/data/'
lig_dir ='/home/juan/Documents/Master_Project/data/correct_pdb/'
cur_dir ='/home/juan/Documents/Master_Project/data/correct_pdb/'
output_file = '/home/juan/Documents/Master_Project/data/'
path_folder ='/home/juan/Documents/Master_Project/data/raw/correct_pdb_ext'

def fname(fp: str):
    return fp.rsplit(".", 1)[0].rsplit("/", 1)[-1]


def get_dists(filename):
    print(filename)
    final_d = np.zeros(len(new_resid_list))
    u = mda.Universe(filename)
    ligand = u.select_atoms('resname UNK')
    ligand_pos = ligand.positions
    u1 = mda.Universe(prot_dir + 'g12d_aligned.pdb')
    for j, res in enumerate(new_resid_list):
        #print(res)
        around_lig = u1.select_atoms(res) 
        
        around_lig_pos = around_lig.positions
        dist_arr = distances.distance_array(ligand_pos, around_lig_pos)
        #print(ligand_pos, around_lig_pos, dist_arr)
        distancesd=[]
        for sublist in dist_arr:
            shortest = np.argmin(sublist)
            #print("Shortest is", shortest, "Value is", sublist[shortest])
            distancesd.append(sublist[shortest])
        short_d = np.min(distancesd)
        final_d[j] = short_d
    return final_d

files = natsorted([fp for fp in os.listdir(cur_dir) if fp.endswith(".pdb") if "g12d" not in fp])


MF = []
error_files = []

for fp in tqdm(files):
    try:
        mol = Chem.MolFromPDBFile(lig_dir + fp)
        if mol is None:
            raise ValueError(f"RDKit failed to create a molecule from file: {fp}")
        fingerprint = AllChem.GetHashedMorganFingerprint(mol, 1, nBits=1024)
        MF.append(fingerprint)
    except Exception as e:
        print(f"An error occurred with file: {fp}")
        print(str(e))
        error_files.append(fp)
        continue

MF_array = np.vstack([np.array(x) for x in MF])

if error_files:
    print("Errors occurred with the following files:", error_files)



files_no_err = [item for item in files if item not in error_files]

# Define a function to process the file and get the distances
def process_file(fp):
    return get_dists(lig_dir + fp)

# Process files and create distances with a single progress bar
with tqdm(total=len(files)) as pbar:
    dists = []
    for fp in files_no_err:
        dists.append(process_file(fp))
        pbar.update(1)
dists = np.stack(dists)
dists = {new_resid_list[i]: dists[:, i] for i in range(dists.shape[1])}


# Re do this to have only the no error files (not necessary)
MF = [AllChem.GetHashedMorganFingerprint(Chem.MolFromPDBFile(lig_dir + fp),
					 1,
					 nBits=1024) for fp in tqdm(files_no_err)]
MF_array = np.vstack([np.array(x.ToList()) for x in MF])

files = [f.replace('.pdb', '') for f in files_no_err]
cols = {f"{i}": MF_array[:,i] for i in range(MF_array.shape[1])}
cols["files"] = files

for k,v in dists.items():
    cols[k] = v



df = pd.DataFrame.from_dict(cols)
df = df.set_index("files")

# Create a new DataFrame with the "files" index turned into a column
new_df = df.reset_index(drop=False)
new_df.rename(columns={'files': 'Folder'}, inplace=True)



barrier_data = pd.read_csv('/home/juan/Documents/Master_Project/QM_Barrier_Data_rbatch.csv')
barrier_data['Folder'] = barrier_data['Folder'].str.strip()
barrier_data.tail()


original_dataset = new_df.merge(barrier_data[['Folder', 'QM/MM SP Barrier']], 
                                on='Folder', 
                                how='left')


n_atoms = []
n_arm_hcycles = []
n_arm_rings = []
n_amidebs = []
sat_carbcs = []
heavy_atoms = []
total_charges = []
folder = []



def count_atoms(path_folder, files):
    data = []
    for file in files:
        file_path = os.path.join(path_folder, file)
        carbon_count = 0
        nitrogen_count = 0
        fluor_count = 0
        oxygen_count = 0
        hydrogen_count = 0
        sulfur_count = 0
        phos_count = 0
        bro_count = 0
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                if line.startswith("ATOM"):
                    elements = line.split()
                    atom_type = elements[2]
                    # Check for the first character of atom_type
                    first_char = atom_type[0]
                    if first_char == "C":
                        carbon_count += 1
                    elif first_char == "H":
                        hydrogen_count += 1
                    elif first_char == "O":
                        oxygen_count += 1
                    elif first_char == "F":
                        fluor_count += 1
                    elif first_char == "N":
                        nitrogen_count += 1
                    elif first_char == "S":
                        sulfur_count += 1
                    elif first_char == "P":
                        phos_count += 1
                    elif atom_type.startswith("Br") or atom_type.startswith("BR"):
                        bro_count += 1
                    else:
                        print(f"Non-C/N/... atom identified in {file}: {atom_type}")
        data.append((file.strip('.pdb'), carbon_count, nitrogen_count, hydrogen_count, oxygen_count, fluor_count, sulfur_count, phos_count, bro_count))

    return pd.DataFrame(data, columns=['Folder', 'C', 'N', 'H', 'O', 'F', 'S', 'P', 'Br'])


for filen in files_no_err:
    # Remove .pdb extension
    filename_without_extension = filen.rstrip('.pdb')
    folder.append(filename_without_extension)

    file_path = os.path.join(path_folder, filen)
    mol = Chem.MolFromPDBFile(file_path)
    if mol:
        n_atom = rdMolDescriptors.CalcNumAtoms(mol)
        n_atoms.append(n_atom)

        heavy_atom = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        heavy_atoms.append(heavy_atom)

        sat_carbc = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
        sat_carbcs.append(sat_carbc)

        n_arm_hcycle = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        n_arm_hcycles.append(n_arm_hcycle)

        n_arm_ring = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_arm_rings.append(n_arm_ring)

        n_amideb = rdMolDescriptors.CalcNumAmideBonds(mol)
        n_amidebs.append(n_amideb)

        # Calculate total charge
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        total_charges.append(total_charge)

    else:
        print(f"Failed to read molecule from {filen}")
data = {
    'atom_number': n_atoms,
    'aromatic_Hcycles': n_arm_hcycles,
    'aromatic_rings': n_arm_rings,
    'amide_bonds': n_amidebs,
    'heavy_atoms':heavy_atoms,
    'saturated_carbocycles': sat_carbcs,
    'total_charge': total_charges,
    'Folder': folder
}

new_feats_df = pd.DataFrame(data)
dataset = pd.merge(original_dataset, new_feats_df, on="Folder", how="inner")


atom_counts_df = count_atoms(path_folder, files)


merge = pd.merge(atom_counts_df, dataset, on='Folder', how='inner')
columns_to_move = ['QM/MM SP Barrier', 'Folder']

all_columns = list(merge.columns)

for column_name in columns_to_move:
    all_columns.remove(column_name)

all_columns.extend(columns_to_move)
merge = merge[all_columns]


electronegativity = {
    "C": 2.55,
    "N": 3.04,
    "H": 2.20,
    "O": 3.44,
    "F": 3.98,
    "S": 2.58,
    "P": 2.19,
    "Br": 2.96}

def weighted_electronegativity(row):
    total_atoms = sum(row[elem] for elem in electronegativity.keys())
    weighted_sum = sum(row[elem] * electronegativity[elem] for elem in electronegativity.keys())
    return weighted_sum / total_atoms if total_atoms > 0 else 0

# Apply the function across the rows
merge['Weighted Electronegativity'] = merge.apply(weighted_electronegativity, axis=1)
cols = ['Weighted Electronegativity'] + [col for col in merge if col != 'Weighted Electronegativity']
merge = merge[cols]


def extract_pdb_features(pdb_file):
    mol = Chem.MolFromPDBFile(pdb_file)
    if mol is None:
        print("Error: Unable to read PDB file.")
        return None
    
    # Calculate the number of bonds
    nBonds = mol.GetNumBonds()
    
    # Initialize the bond type counts
    nBonds2 = 0
    nBondsD = 0
    nBondsT = 0
    nBondsM = 0
    
    # Loop through each bond to count the types
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            nBonds2 += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            nBondsD += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            nBondsT += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            nBondsM += 1
    # Calculate counts of rings of different sizes
    ri = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ri.AtomRings()]
    nRing = len(ring_sizes)
    n3Ring = ring_sizes.count(3)
    n4Ring = ring_sizes.count(4)
    n5Ring = ring_sizes.count(5)
    n6Ring = ring_sizes.count(6)
    n7Ring = ring_sizes.count(7)
    n8Ring = ring_sizes.count(8)
    nMoreRing = sum(1 for size in ring_sizes if size > 8)

    # Initialize counts for rings containing heteroatoms
    heteroatom_ring_counts = {
        'Total Heteroatom Rings': 0,
        '3-Membered Heteroatom Rings': 0,
        '4-Membered Heteroatom Rings': 0,
        '5-Membered Heteroatom Rings': 0,
        '6-Membered Heteroatom Rings': 0,
        '7-Membered Heteroatom Rings': 0,
        '12-Membered Heteroatom Rings': 0,
        'More than 12-Membered Heteroatom Rings': 0
    }

    # Ensure that Chem.GetSymmSSSR(mol) returns ring objects
    ring_objs = list(Chem.GetSymmSSSR(mol))
    ring_objs = [list(vec) for vec in ring_objs]

    if isinstance(ring_objs, int):
        # No rings found, set ring-related counts to 0
        nRing = 0
    else:
        # Iterate over each ring in the molecule
        # Iterate over each ring in the molecule
        for ring in ring_objs:
            # Check if the ring contains heteroatoms
            if any(mol.GetAtomWithIdx(atom).GetAtomicNum() != 6 for atom in ring):
                heteroatom_ring_counts['Total Heteroatom Rings'] += 1

                # Categorize the heteroatom-containing rings based on the number of atoms they contain
                num_atoms = len(ring)
                if num_atoms == 3:
                    heteroatom_ring_counts['3-Membered Heteroatom Rings'] += 1
                elif num_atoms == 4:
                    heteroatom_ring_counts['4-Membered Heteroatom Rings'] += 1
                elif num_atoms == 5:
                    heteroatom_ring_counts['5-Membered Heteroatom Rings'] += 1
                elif num_atoms == 6:
                    heteroatom_ring_counts['6-Membered Heteroatom Rings'] += 1
                elif num_atoms == 7:
                    heteroatom_ring_counts['7-Membered Heteroatom Rings'] += 1
                elif num_atoms == 12:
                    heteroatom_ring_counts['12-Membered Heteroatom Rings'] += 1
                else:
                    heteroatom_ring_counts['More than 12-Membered Heteroatom Rings'] += 1

    # Return the extracted features
    features = {
        'nBonds': nBonds,
        'nBondsD': nBondsD,
        'nBonds2': nBonds2,
        'nBondsT': nBondsT,
        'nBondsM': nBondsM,
        'pubchemcomplexity': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        'pubchemhbondacceptor': rdMolDescriptors.CalcNumLipinskiHBA(mol),
        'pubchemhbonddonor': rdMolDescriptors.CalcNumLipinskiHBD(mol),
        'pubchemrotbon': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'pubchemlogp': Crippen.MolLogP(mol),
        'pubchemmass': Descriptors.MolWt(mol),
        'pubchemtsa': rdMolDescriptors.CalcTPSA(mol),
        'nRing': nRing,
        'n3Ring': n3Ring,
        'n4Ring': n4Ring,
        'n5Ring': n5Ring,
        'n6Ring': n6Ring,
        'n7Ring': n7Ring,
        'n8Ring': n8Ring,
        'nMoreRing': nMoreRing,
        **heteroatom_ring_counts
    }
    return features
        

# Function to extract features from all PDB files in a given directory and return a DataFrame
def extract_features_from_directory(directory_path):
    features_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdb"):
            file_path = os.path.join(directory_path, filename)
            features = extract_pdb_features(file_path)
            if features is not None:
                features['Folder'] = filename.strip(".pdb")  # Add the filename to the features
                features_list.append(features)
    
    return pd.DataFrame(features_list)

