import numpy as np
import re
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
folder_path1 = '/path/to/pdbqt_or_pdb/folder'
folder_path2 = '/path/to/psf/folder'

def parse_psf_charge_data(psf_file_path):
    charges = defaultdict(float) 
    parsing_atoms = False
    with open(psf_file_path, 'r') as file:
        for line in file:
            # Check for the start of the atom section
            if "!NATOM" in line:
                parsing_atoms = True
                continue
            
            if parsing_atoms:
                if line.strip().isdigit() or not line.strip():
                    parsing_atoms = False  # End of atom section
                    continue
                
                parts = re.findall(r'\S+', line)
                if len(parts) >= 6:
                    atom_id = int(parts[0])  # Unique atom identifier
                    atom_name = parts[4] 
                    charge = float(parts[6])
                    charges[atom_id] = (atom_name, charge)  # Store atom name and charge by atom ID
    return charges

def parse_pdbqt_atom_data(pdbqt_file_path):
    atoms = []
    with open(pdbqt_file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                atom_type = parts[2][0]  # Simplifying to first letter might not always be accurate for pdbqt
                x = float(parts[6])  # Adjusted indices for pdbqt format
                y = float(parts[7])
                z = float(parts[8])
                atoms.append((atom_type, x, y, z))
    return atoms


def process_files(pdbqt_folder_path, psf_folder_path):
    combined_data = defaultdict(list)
    
    pdbqt_files = [file for file in os.listdir(pdbqt_folder_path) if file.endswith(".pdb")] # pdbqt or pdb
    psf_files = [file for file in os.listdir(psf_folder_path) if file.endswith(".psf")]

    # Ensure processing of matching PDBQT and PSF files
    for pdbqt_file in pdbqt_files:
        base_name = os.path.splitext(pdbqt_file)[0]  # mol_number without extension
        psf_file_name = base_name + "_new.psf"  # Corresponding PSF file
        
        if psf_file_name in psf_files:
            pdbqt_path = os.path.join(pdbqt_folder_path, pdbqt_file)
            psf_path = os.path.join(psf_folder_path, psf_file_name)
            
            pdbqt_atoms = parse_pdbqt_atom_data(pdbqt_path)
            psf_charges = parse_psf_charge_data(psf_path)

            for atom_index, atom in enumerate(pdbqt_atoms, start=1):  
                atom_type, x, y, z = atom
                psf_data = psf_charges.get(atom_index, ("Unknown", 0.0)) 
                _, charge = psf_data  # Unpack atom name and charge

                combined_data['File Name'].append(pdbqt_file)
                combined_data['Atom Type'].append(atom_type)
                combined_data['X'].append(x)
                combined_data['Y'].append(y)
                combined_data['Z'].append(z)
                combined_data['Charge'].append(charge)

    return pd.DataFrame(combined_data)

df = process_files(folder_path1, folder_path2)

# Box Dimension
box_x_updated = 6 # Same along xyz dimension

# Ranges
x_range_updated = 6 - (-18)  
y_range_updated = 19 - (-11) 
z_range_updated = 13 - (-11)


num_boxes_x_updated_new = x_range_updated // box_x_updated
num_boxes_y_updated_new = y_range_updated // box_x_updated
num_boxes_z_updated_new = z_range_updated // box_x_updated

total_boxes_updated_again = num_boxes_x_updated_new * num_boxes_y_updated_new * num_boxes_z_updated_new


def box_coverage(num_boxes_x, num_boxes_y, num_boxes_z, box_x, box_y, box_z, start_x=-18, start_y=-11, start_z=-11):
    boxes = []
    for z in range(num_boxes_z):
        for y in range(num_boxes_y):
            for x in range(num_boxes_x):
                box_start_x = start_x + x * box_x
                box_end_x = box_start_x + box_x
                box_start_y = start_y + y * box_y
                box_end_y = box_start_y + box_y
                box_start_z = start_z + z * box_z
                box_end_z = box_start_z + box_z
                boxes.append({
                    "box_number": len(boxes) + 1,
                    "x_range": (box_start_x, box_end_x),
                    "y_range": (box_start_y, box_end_y),
                    "z_range": (box_start_z, box_end_z),
                })
    return boxes

# Generate coverage for each box with the dimensions
box_list = box_coverage(num_boxes_x_updated_new, num_boxes_y_updated_new, num_boxes_z_updated_new, box_x_updated, box_x_updated, box_x_updated)

def assign_atoms_to_boxes_for_df(df, box_list):
    # Initialize structure for total charge per box, per file
    file_box_charges = defaultdict(lambda: {box['box_number']: 0.0 for box in box_list})

    def find_box_for_atom(x, y, z, box_list):
        for box in box_list:
            if box['x_range'][0] <= x < box['x_range'][1] and \
               box['y_range'][0] <= y < box['y_range'][1] and \
               box['z_range'][0] <= z < box['z_range'][1]:
                return box['box_number']
        return None

    for atom in df.itertuples(index=False):
        file_name = atom._0
        x, y, z, charge = atom.X, atom.Y, atom.Z, atom.Charge
        box_number = find_box_for_atom(x, y, z, box_list)

        if box_number is not None:
            file_box_charges[file_name][box_number] += charge

    return dict(file_box_charges)

box_charge = assign_atoms_to_boxes_for_df(df, box_list)

def convert_box_charge_to_df(box_charge):
    # Prepare a list to collect rows for the DataFrame
    rows_list = []

    # Iterate through each file in the box_charge dictionary
    for file_name, charges in box_charge.items():
        # Start building a row with file name
        row = {'File Name': file_name}
        # Add charges for each box to the row
        for box_number, charge in charges.items():
            row[f'Box_{box_number}'] = charge
        # Append the row to the list
        rows_list.append(row)
    
    # Create the DataFrame
    df = pd.DataFrame(rows_list)
    
    # If some boxes might be missing in some files, fill their values with 0.0
    # Ensure all expected boxes are present as columns
    expected_boxes = [f'Box_{i}' for i in range(1, 81)]
    for box in expected_boxes:
        if box not in df:
            df[box] = 0.0
    
    # Reorder DataFrame columns to start with 'File Name' followed by box columns
    df = df[['File Name'] + expected_boxes]
    
    return df
