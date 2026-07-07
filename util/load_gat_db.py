import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from rdkit import Chem

from util import osUtils as ou


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def readKGData(path="data_set/DrugCombDB/kg_final2.txt"):
    print("Read knowledge graph data...")
    entity_set = set()
    relation_set = set()
    triples = []
    for h, r, t in ou.readTriple(path, sep=","):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])
    return list(entity_set), list(relation_set), triples


def readRecData(path="data_set/DrugCombDB/comb_final_filtered.txt", test_ratio=0.2):
    print("Read drug combination synergy data...")
    drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, cell, label, fold in ou.readTriple(path, sep=","):
        drug_set1.add(int(d1))
        drug_set2.add(int(d2))
        cell_set.add(int(cell))
        triples.append((int(d1), int(d2), int(cell), int(label), int(fold)))
    return list(drug_set1), list(drug_set2), list(cell_set), triples


def load_smiles_data(file_path):
    df = pd.read_csv(file_path)
    molecule_data = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue
        molecule_data.append(molecule_to_graph(mol))
    return molecule_data


def molecule_to_graph(molecule):
    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)
    atom_features = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    atom_features = torch.tensor(atom_features, dtype=torch.float32)
    return adjacency_matrix, atom_features
