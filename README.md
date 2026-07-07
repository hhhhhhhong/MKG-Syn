<<<<<<< HEAD
# mkg_syn

`mkg_syn` is a multi-modal knowledge graph model for drug combination synergy prediction. It uses drug pairs, cell lines, knowledge graph information, SMILES features, text embeddings, and gene expression features to predict whether a drug combination is synergistic.

## Structure

```text
.
|-- configs/
|   |-- config.yaml
|   `-- config_db.yaml
|-- data_set/
|-- models/
|-- os_saved_features/
|-- saved_features/
|-- subgragh/
|-- util/
|-- model.py
|-- train.py
|-- train_cl.py
|-- train_cl_db.py
`-- requirements.txt
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch Geometric extension wheels are missing, install the wheel set that matches your PyTorch and CUDA version. For example, with PyTorch 2.2 and CUDA 11.8:

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

## Required Files

Run scripts from the project root. Paths are configured relative to the root directory.

### Download Models

Download the two required Hugging Face models before running the pipeline:

- ChemBERTa: [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
- PubMedBERT embeddings: [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)

Recommended local layout:

```bash
mkdir -p models
git lfs install
git clone https://huggingface.co/DeepChem/ChemBERTa-77M-MLM models/ChemBERTa-77M-MLM
git clone https://huggingface.co/NeuML/pubmedbert-base-embeddings models/pubmedbert-base-embeddings
```

`ChemBERTa-77M-MLM` is used by the training scripts. `pubmedbert-base-embeddings` is used by the `subgragh/` embedding scripts.

### Data

Expected dataset layout:

```text
data_set/
|-- OncologyScreen/
|   |-- comb_final.txt
|   |-- kg_final2.txt
|   |-- smiles_completed.csv
|   `-- cell_gene.csv
`-- DrugCombDB/
    |-- comb_final_filtered.txt
    |-- kg_final2.txt
    |-- smiles_completed.csv
    `-- ccle_cell_gene.csv
```

### Text Embeddings

OncologyScreen:

```text
os_saved_features/
|-- drug_global_embeddings.pt
`-- cell_global_embeddings.pt
```

DrugCombDB:

```text
saved_features/
|-- drug_global_embeddings.pt
`-- cell_global_embeddings.pt
```

### Local Models

Default model paths:

```text
models/
|-- ChemBERTa-77M-MLM/
`-- pubmedbert-base-embeddings/
```

## Pipeline

The normal workflow is:

```text
1. Download ChemBERTa and PubMedBERT from Hugging Face
2. Run scripts under subgragh/ to generate text embeddings
3. Run train.py
```

Generate DrugCombDB text embeddings:

```bash
python subgragh/step1_extract_subgraphs.py --dataset DrugCombDB
python subgragh/step2_extract_embeddings.py --input-dir saved_features
```

This generates:

```text
saved_features/
|-- drug_global_embeddings.pt
`-- cell_global_embeddings.pt
```

Generate OncologyScreen text embeddings:

```bash
python subgragh/step1_extract_subgraphs.py --dataset OncologyScreen --output-dir os_saved_features
python subgragh/step2_extract_embeddings.py --input-dir os_saved_features
```

This generates:

```text
os_saved_features/
|-- drug_global_embeddings.pt
`-- cell_global_embeddings.pt
```

## Training

Train on OncologyScreen:

```bash
python train.py --config-name config
```

Train on DrugCombDB:

```bash
python train.py --config-name config_db
```

Before running DrugCombDB training, make sure `data_set/DrugCombDB/comb_final_filtered.txt` exists. The `subgragh/` scripts generate text embeddings only; they do not generate the training label file.

Select GPU from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name config_db
```

Override config values with Hydra:

```bash
python train.py --config-name config_db training.batch_size=128 training.n_epochs=200
```

Change paths without editing code:

```bash
python train.py --config-name config_db \
  paths.chemberta_dir=/path/to/ChemBERTa-77M-MLM \
  paths.drug_emb=/path/to/drug_global_embeddings.pt \
  paths.cell_emb=/path/to/cell_global_embeddings.pt
```

The old commands still work as compatibility shortcuts:

```bash
python train_cl.py
python train_cl_db.py
```

## Configs

Main config files:

```text
configs/config.yaml
configs/config_db.yaml
```

Important path fields:

```yaml
paths:
  root: "${hydra:runtime.cwd}/data_set"
  rec_path: "${paths.root}/..."
  kg_path: "${paths.root}/..."
  smiles_csv: "${paths.root}/..."
  drug_emb: "${hydra:runtime.cwd}/..."
  cell_emb: "${hydra:runtime.cwd}/..."
  chemberta_dir: "${hydra:runtime.cwd}/models/ChemBERTa-77M-MLM"
```

Outputs are saved under:

```text
outputs/
checkpoints/
```

## Notes

- Keep large generated files out of GitHub: `models/`, `*.pt`, `checkpoints/`, `outputs/`, `wandb/`.
- If datasets or embeddings cannot be redistributed, provide download or generation instructions instead.
- Add a `LICENSE` file before public release.
=======
# MKG-Syn
>>>>>>> f8e258de18290a4691325b203d899eccc8417238
