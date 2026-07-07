import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REL_DRUG_TARGET = 0
REL_CELL_PROTEIN = 1
REL_CELL_TISSUE = 2
REL_PROTEIN_PROTEIN = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Extract drug and cell subgraph texts.")
    parser.add_argument("--dataset", default="DrugCombDB", choices=["DrugCombDB", "OncologyScreen"])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to saved_features for DrugCombDB and os_saved_features for OncologyScreen.",
    )
    return parser.parse_args()


def resolve_output_dir(dataset, output_dir):
    if output_dir is None:
        output_dir = "saved_features" if dataset == "DrugCombDB" else "os_saved_features"

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root


def resolve_paths(dataset, output_dir):
    data_root = PROJECT_ROOT / "data_set" / dataset
    output_root = resolve_output_dir(dataset, output_dir)
    return {
        "mapping_file": data_root / "id2name_translated.json",
        "triplets_file": data_root / "kg_final2.txt",
        "drug_csv": data_root / "drug_id.csv",
        "cell_csv": data_root / "cell_id.csv",
        "output_dir": output_root,
    }


def load_mapping(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_entity_name(raw_name):
    return (
        raw_name.replace("(Drug)", "")
        .replace("(Protein)", "")
        .replace("(Cell)", "")
        .replace("(Tissue)", "")
        .strip()
    )


def build_knowledge_graph(path):
    print(f"Building undirected knowledge graph from: {path}")
    graph = nx.Graph()
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                head, relation, tail = map(int, parts[:3])
            except ValueError:
                continue
            graph.add_edge(head, tail, relation=relation)

    print(f"Graph ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def read_unique_ids(path, column):
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"{path} must contain column: {column}")
    return [int(value) for value in df[column].dropna().unique()]


def extract_drug_global_subgraph(graph, drug_id, top_k_targets=15, top_k_downstream=20):
    unique_triplets = set()
    targets = []

    if drug_id in graph:
        for neighbor in graph.neighbors(drug_id):
            if graph[drug_id][neighbor]["relation"] == REL_DRUG_TARGET:
                targets.append(neighbor)

    if len(targets) > top_k_targets:
        targets = sorted(targets, key=lambda node: graph.degree(node), reverse=True)[:top_k_targets]

    for target in targets:
        unique_triplets.add((drug_id, REL_DRUG_TARGET, target))

    downstream_nodes = []
    for target in targets:
        for second_neighbor in graph.neighbors(target):
            if second_neighbor == drug_id:
                continue
            relation = graph[target][second_neighbor]["relation"]
            if relation in {REL_DRUG_TARGET, REL_CELL_PROTEIN, REL_PROTEIN_PROTEIN}:
                downstream_nodes.append((target, relation, second_neighbor))

    for triplet in downstream_nodes[:top_k_downstream]:
        unique_triplets.add(triplet)

    return unique_triplets


def serialize_drug_subgraph(unique_triplets, id2name):
    if not unique_triplets:
        return "Unknown drug mechanism."

    edges_desc = []
    sorted_triplets = sorted(unique_triplets, key=lambda item: (item[0], item[2]))
    for head, relation, tail in sorted_triplets:
        raw_head = id2name.get(str(head), "")
        raw_tail = id2name.get(str(tail), "")
        name_head = clean_entity_name(raw_head)
        name_tail = clean_entity_name(raw_tail)

        if relation == REL_DRUG_TARGET:
            if "(Drug)" in raw_head:
                edges_desc.append(f"drug {name_head} targets protein {name_tail}")
            else:
                edges_desc.append(f"protein {name_head} is targeted by drug {name_tail}")
        elif relation == REL_PROTEIN_PROTEIN:
            edges_desc.append(f"protein {name_head} interacts with protein {name_tail}")
        elif relation == REL_CELL_PROTEIN:
            if "(Protein)" in raw_head:
                edges_desc.append(f"protein {name_head} is associated with cell line {name_tail}")
            else:
                edges_desc.append(f"cell line {name_head} expresses protein {name_tail}")
        elif relation == REL_CELL_TISSUE:
            edges_desc.append(f"cell line {name_head} belongs to tissue {name_tail}")

    return "The biological interaction network is formed by the following mechanisms: " + "; ".join(edges_desc) + "."


def extract_cell_ego_subgraph(graph, cell_id, top_k_proteins=20):
    unique_triplets = set()
    connected_proteins = []

    if cell_id in graph:
        for neighbor in graph.neighbors(cell_id):
            relation = graph[cell_id][neighbor]["relation"]
            if relation == REL_CELL_TISSUE:
                unique_triplets.add((cell_id, REL_CELL_TISSUE, neighbor))
            elif relation == REL_CELL_PROTEIN:
                connected_proteins.append(neighbor)

    if len(connected_proteins) > top_k_proteins:
        connected_proteins = sorted(connected_proteins, key=lambda node: graph.degree(node), reverse=True)[:top_k_proteins]

    for protein in connected_proteins:
        unique_triplets.add((protein, REL_CELL_PROTEIN, cell_id))

    for left_idx in range(len(connected_proteins)):
        for right_idx in range(left_idx + 1, len(connected_proteins)):
            protein_1 = connected_proteins[left_idx]
            protein_2 = connected_proteins[right_idx]
            if graph.has_edge(protein_1, protein_2) and graph[protein_1][protein_2]["relation"] == REL_PROTEIN_PROTEIN:
                unique_triplets.add((protein_1, REL_PROTEIN_PROTEIN, protein_2))

    return unique_triplets


def serialize_cell_subgraph(unique_triplets, id2name):
    if not unique_triplets:
        return "Unknown cell line microenvironment."

    tissue_desc = ""
    protein_desc = []
    ppi_desc = []

    for head, relation, tail in unique_triplets:
        name_head = clean_entity_name(id2name.get(str(head), ""))
        name_tail = clean_entity_name(id2name.get(str(tail), ""))

        if relation == REL_CELL_TISSUE:
            tissue_desc = f"The cell line {name_head} belongs to tissue {name_tail}."
        elif relation == REL_CELL_PROTEIN:
            protein_desc.append(name_head)
        elif relation == REL_PROTEIN_PROTEIN:
            ppi_desc.append(f"{name_head} interacts with {name_tail}")

    text_parts = []
    if tissue_desc:
        text_parts.append(tissue_desc)
    if protein_desc:
        text_parts.append(f"It expresses key functional proteins including {', '.join(protein_desc)}.")
    if ppi_desc:
        text_parts.append("Within this cell, the active protein interaction network includes: " + "; ".join(ppi_desc) + ".")

    return " ".join(text_parts)


if __name__ == "__main__":
    args = parse_args()
    paths = resolve_paths(args.dataset, args.output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    id2name = load_mapping(paths["mapping_file"])
    graph = build_knowledge_graph(paths["triplets_file"])

    unique_drugs = read_unique_ids(paths["drug_csv"], "drug_id")
    unique_cells = read_unique_ids(paths["cell_csv"], "cell_id")

    cell_texts = {}
    drug_texts = {}

    print(f"\n[1/2] Extracting texts for {len(unique_cells)} cell lines...")
    for cell_id in tqdm(unique_cells):
        triplets = extract_cell_ego_subgraph(graph, cell_id)
        cell_texts[cell_id] = serialize_cell_subgraph(triplets, id2name)

    print(f"\n[2/2] Extracting texts for {len(unique_drugs)} drugs...")
    for drug_id in tqdm(unique_drugs):
        triplets = extract_drug_global_subgraph(graph, drug_id)
        drug_texts[drug_id] = serialize_drug_subgraph(triplets, id2name)

    with (paths["output_dir"] / "cell_texts.json").open("w", encoding="utf-8") as file:
        json.dump(cell_texts, file, indent=4)

    with (paths["output_dir"] / "drug_texts.json").open("w", encoding="utf-8") as file:
        json.dump(drug_texts, file, indent=4)

    print(f"\nText extraction complete. Saved to: {paths['output_dir']}")
