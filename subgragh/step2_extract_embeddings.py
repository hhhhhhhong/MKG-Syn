import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "pubmedbert-base-embeddings")
DEFAULT_FEATURE_DIR = str(PROJECT_ROOT / "saved_features")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode extracted subgraph texts with PubMedBERT.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local PubMedBERT model directory.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing cell_texts.json and drug_texts.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated .pt embeddings. Defaults to --input-dir.",
    )
    return parser.parse_args()


class TextEncoder:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PubMedBERT on {self.device} from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    def encode(self, text):
        if not text or text in {"Unknown drug mechanism.", "Unknown cell line microenvironment."}:
            return torch.zeros(self.hidden_size, dtype=torch.float32)

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
        return embedding


def load_texts(input_dir, file_name):
    path = Path(input_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = TextEncoder(args.model_path)

    cell_texts_dict = load_texts(input_dir, "cell_texts.json")
    drug_texts_dict = load_texts(input_dir, "drug_texts.json")

    cell_embeddings_dict = {}
    drug_embeddings_dict = {}

    print(f"\n[1/2] Encoding {len(cell_texts_dict)} cell-line texts...")
    for cell_id_str, text in tqdm(cell_texts_dict.items()):
        cell_embeddings_dict[int(cell_id_str)] = encoder.encode(text)

    print(f"\n[2/2] Encoding {len(drug_texts_dict)} drug texts...")
    for drug_id_str, text in tqdm(drug_texts_dict.items()):
        drug_embeddings_dict[int(drug_id_str)] = encoder.encode(text)

    cell_output = output_dir / "cell_global_embeddings.pt"
    drug_output = output_dir / "drug_global_embeddings.pt"
    torch.save(cell_embeddings_dict, cell_output)
    torch.save(drug_embeddings_dict, drug_output)

    print("\nEmbedding extraction complete.")
    print(f"Saved: {cell_output}")
    print(f"Saved: {drug_output}")
