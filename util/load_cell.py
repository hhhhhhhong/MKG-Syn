import pandas as pd
import torch


def get_gene_matrix_tensor(mapping_path, expression_path, n_entitys):
    map_df = pd.read_csv(mapping_path)
    id2model = dict(zip(map_df["瀹炰綋id"], map_df["modelid"]))

    expr_df = pd.read_csv(expression_path, index_col=0)
    meta_cols = ["SequencingID", "ModelID", "IsDefaultEntryForModel", "ModelConditionID", "IsDefaultEntryForMC"]
    expr_df = expr_df.drop(columns=meta_cols, errors="ignore")

    num_genes = expr_df.shape[1]
    matrix = torch.zeros(n_entitys, num_genes, dtype=torch.float32)

    for ent_id, model_id in id2model.items():
        if ent_id < n_entitys and model_id in expr_df.index:
            matrix[ent_id] = torch.tensor(expr_df.loc[model_id].values, dtype=torch.float32)

    return matrix
