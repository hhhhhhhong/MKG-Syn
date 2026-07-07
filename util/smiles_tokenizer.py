import numpy as np
import pandas as pd
import torch


CHARCANSMISET = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9,
    "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
    "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25, "I": 26, "H": 27,
    "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
    "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, "]": 43, "_": 44, "a": 45,
    "c": 46, "b": 47, "e": 48, "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
    "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60, "t": 61, "y": 62, "@": 63,
    "/": 64, "\\": 0
}
VOCAB_SIZE = max(CHARCANSMISET.values()) + 1
SMI_MAX_LEN = 100


def smiles_to_tensor(csv_path: str, entity_num: int) -> torch.LongTensor:
    df = pd.read_csv(csv_path)
    tensor = np.zeros((entity_num, SMI_MAX_LEN), dtype=np.int64)
    for _, row in df.iterrows():
        eid = int(row.entity_id)
        smiles = str(row.smiles)[:SMI_MAX_LEN]
        tokens = [CHARCANSMISET.get(char, 1) for char in smiles]
        tensor[eid, :len(tokens)] = tokens
    return torch.LongTensor(tensor)
