import collections

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def construct_kg(kgTriples):
    print("Building knowledge graph index...")
    kg = {}
    for head, relation, tail in kgTriples:
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def getKgIndexsFromKgTriples(kg_triples):
    kg_indexs = collections.defaultdict(list)
    for h, r, t in kg_triples:
        kg_indexs[str(h)].append([int(t), int(r)])
    return kg_indexs


def filetDateSet(dataSet, user_pos):
    return [item for item in dataSet if str(item[0]) in user_pos]


def construct_adj(neighbor_sample_size, kg_indexes, entity_num):
    print("Building entity and relation adjacency matrices...")
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg_indexes[str(entity)]
        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            continue
        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)),
                size=neighbor_sample_size,
                replace=False
            )
        else:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)),
                size=neighbor_sample_size,
                replace=True
            )
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation


def construct_molecule_graphs(smiles_csv_path):
    print("Building molecule graph features...")
    df = pd.read_csv(smiles_csv_path)

    drug_graph_features = {}
    drug_adj_matrices = {}

    for _, row in df.iterrows():
        drug_id = int(row["entity_id"]) - 1
        smiles = row["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue

        atom_features = []
        for atom in mol.GetAtoms():
            valence = atom.GetExplicitValence()
            try:
                valence += atom.GetImplicitValence()
            except Exception:
                pass

            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                valence,
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic())
            ])
        atom_features = torch.tensor(atom_features, dtype=torch.float)

        n_atoms = mol.GetNumAtoms()
        adj = np.zeros((n_atoms, n_atoms), dtype=float)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = torch.tensor(adj, dtype=torch.float)

        drug_graph_features[drug_id] = atom_features
        drug_adj_matrices[drug_id] = adj

    return drug_graph_features, drug_adj_matrices
