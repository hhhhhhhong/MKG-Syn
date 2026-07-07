import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset


num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3


def atom_to_feature(atom):
    atomic_num = atom.GetAtomicNum()
    atom_type = min(atomic_num, num_atom_type - 1)

    chiral_tag = atom.GetChiralTag()
    if chiral_tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        chirality = 0
    elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chirality = 1
    else:
        chirality = 2

    return [atom_type, chirality]


def bond_to_feature(bond):
    bond_type_raw = bond.GetBondType()
    if bond_type_raw == Chem.rdchem.BondType.SINGLE:
        bond_type = 0
    elif bond_type_raw == Chem.rdchem.BondType.DOUBLE:
        bond_type = 1
    elif bond_type_raw == Chem.rdchem.BondType.TRIPLE:
        bond_type = 2
    elif bond_type_raw == Chem.rdchem.BondType.AROMATIC:
        bond_type = 3
    else:
        bond_type = 4

    bond_dir_raw = bond.GetBondDir()
    if bond_dir_raw == Chem.rdchem.BondDir.NONE:
        bond_dir = 0
    elif bond_dir_raw == Chem.rdchem.BondDir.BEGINWEDGE:
        bond_dir = 1
    else:
        bond_dir = 2

    return [bond_type, bond_dir]


class SmilesDataset(Dataset):
    def __init__(self, csv_path, smiles_column="smiles", transform=None):
        super().__init__(None, transform)
        self.df = pd.read_csv(csv_path)
        self.smiles_list = self.df[smiles_column].tolist()

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            x = torch.zeros((1, 2), dtype=torch.long)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=idx)

        atom_feats = [atom_to_feature(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.long)

        edge_index_list = []
        edge_attr_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feature = bond_to_feature(bond)

            edge_index_list.append([i, j])
            edge_attr_list.append(bond_feature)
            edge_index_list.append([j, i])
            edge_attr_list.append(bond_feature)

        if len(edge_index_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            idx=torch.tensor([idx])
        )
