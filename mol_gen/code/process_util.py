import logging
import os
import tarfile

import torch
from torch.nn.utils.rnn import pad_sequence

#映射原子与电荷量
charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, "S": 16, "Cl": 17}



def split_dataset(data,split_ids):
   
    split_data = {}
    for set, split in split_ids.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data


def process_qm9(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        def readfile(data_pt):
            return tardata.extractfile(data_pt)

    elif os.path.isdir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        def readfile(data_pt):
            return open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in
                     molecules.items()}

    return molecules




def process_xyz_qm9(datafile):
    
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms + 2]
    mol_freq = xyz_lines[num_atoms + 2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                    'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule
   def is_int(str):
    try:
        int(str)
        return True
    except:
        return False


# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass




