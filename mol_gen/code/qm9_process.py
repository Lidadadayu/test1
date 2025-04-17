import logging
import os
import urllib
import urllib.request
from os.path import join as join

import numpy as np
import pandas as pd

# 假设这些函数已经定义
from process_util import process_qm9, process_xyz_qm9
from utils import is_int, cleanup_file

def process_dataset_qm9(data_dir, data_name):
   
    qm9_tar_data = join(*[data_dir, data_name])
    print(qm9_tar_data)

    print("now processing qm9 dataset:",qm9_tar_data)

    #p判断数据集是否存在
    if not os.path.exists(qm9_tar_data):
        raise FileNotFoundError(f"未找到数据集文件: {qm9_tar_data}")
     
     #分割数据集，得到索引列表
    splits = gen_splits_qm9()
    print(splits.keys())
 
    qm9_data = {}
    for split, split_idx in splits.items():
        qm9_data[split] = process_qm9(
            qm9_tar_data, process_xyz_qm9, file_idx_list=split_idx, stack=True)

    # Download thermochemical energy from QM9 dataset, and then process it into a dictionary
    therm_energy = get_thermo_dict(data_dir)

    # For each of train/validation/test split, add the thermochemical energy
    for split_idx, split_data in qm9_data.items():
        qm9_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed QM9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for key, value in qm9_data.items():
        print(f"Processing {key} data...")
        if isinstance(value, np.ndarray):
            # 将数组转换为 DataFrame
            df = pd.DataFrame(value)
            # 修改此处，使用 key 作为文件名的一部分
            savedir = join(data_dir, f'{key}.csv')
            print(f"Saving data to {savedir}")  # 添加调试信息
            df.to_csv(savedir, index=False)

    logging.info(f'successfully save csv data in folder:{data_dir}!')


def gen_splits_qm9():
    data_dir ='E:\\pycode\\mol_gen\\data\\raw_data\\qm9'
    qm9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    qm9_txt_excluded = join(data_dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(qm9_url_excluded, filename=qm9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    print('Reading excluded indices from file uncharacterized.txt')
    with open(qm9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # Now, create a list of indices
    qm9_num = 133885
    Nexcluded_num = 3054

    included_idxs = np.array(
        sorted(list(set(range(qm9_num)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = qm9_num - Nexcluded_num

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain + Nvalid, Ntrain + Nvalid + Ntest])

    #assert (len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
       # len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    print('len(train):', len(train), 'len(valid):', len(valid), 'len(test):', len(test))
    splits = {'train': train, 'valid': valid, 'test': test}

    return splits


def get_thermo_dict(data_dir):
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    qm9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    qm9_txt_thermo = join(data_dir, 'atomref.txt')

    urllib.request.urlretrieve(qm9_url_thermo, filename=qm9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(qm9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                ] = float(split_therm)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data

 
def get_unique_charges(charges):
    """
    函数通过遍历所有分子的电荷信息，
    统计了每个分子中不同电荷的出现次数，并将结果存储在一个字典中返回。
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts


if __name__ == '__main__':
    data_dir = 'E:\\pycode\\mol_gen\\data\\raw_data\\qm9'
    data_name = 'dsgdb9nsd.xyz.tar.bz2'
    process_dataset_qm9(data_dir, data_name)