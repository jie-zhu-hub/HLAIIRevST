import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


REQUIRED_COLUMNS = ["CDR3", "HLA_type", "MT_pep", "HLA_alpha", "HLA_beta"]
INVALID_AA = set("XUOB")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def standardize_positive_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {}
    if "Label" in df.columns and "label" not in df.columns:
        rename_map["Label"] = "label"
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep_cols = REQUIRED_COLUMNS + [c for c in ["label", "peplen"] if c in df.columns]
    df = df[keep_cols].copy()

    if "label" not in df.columns:
        df["label"] = 1
    else:
        df["label"] = 1

    df["peplen"] = df["MT_pep"].astype(str).str.len()
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    df = df.drop_duplicates(subset=["CDR3", "MT_pep", "HLA_type"]).reset_index(drop=True)
    return df


def filter_invalid_sequences(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["CDR3", "MT_pep"]:
        out = out[out[col].astype(str).map(lambda x: not any(ch in INVALID_AA for ch in x))]
    out = out.dropna(subset=REQUIRED_COLUMNS + ["label", "peplen"]).reset_index(drop=True)
    return out


def build_hla_map(df: pd.DataFrame) -> dict:
    hla_map = {}
    for _, row in df[["HLA_type", "HLA_alpha", "HLA_beta"]].drop_duplicates().iterrows():
        hla = row["HLA_type"]
        if hla not in hla_map:
            hla_map[hla] = (row["HLA_alpha"], row["HLA_beta"])
    return hla_map


def random_shuffle(pos_df: pd.DataFrame, num: int) -> pd.DataFrame:
    peptides = pos_df["MT_pep"].tolist()
    hla_types = pos_df["HLA_type"].tolist()
    hla_map = build_hla_map(pos_df)

    neg_rows = []
    for i in range(num):
        for _, row in pos_df.iterrows():
            cdr3 = row["CDR3"]
            pos_pep = row["MT_pep"]
            pos_hla = row["HLA_type"]

            neg_pep = random.choice(peptides)
            neg_hla = random.choice(hla_types)
            while neg_pep == pos_pep and neg_hla == pos_hla:
                neg_pep = random.choice(peptides)
                neg_hla = random.choice(hla_types)

            hla_alpha, hla_beta = hla_map[neg_hla]
            neg_rows.append({
                "CDR3": cdr3,
                "HLA_type": neg_hla,
                "MT_pep": neg_pep,
                "HLA_alpha": hla_alpha,
                "HLA_beta": hla_beta,
                "label": 0,
                "peplen": len(str(neg_pep)),
            })

    neg_df = pd.DataFrame(neg_rows)
    neg_df = neg_df.drop_duplicates(subset=["CDR3", "MT_pep", "HLA_type"]).reset_index(drop=True)
    return neg_df


def _sample_items(candidates, k: int):
    if len(candidates) == 0:
        raise ValueError("The negative-sampling candidate set is empty; unipep sampling cannot continue.")
    replace = len(candidates) < k
    return np.random.choice(list(candidates), size=k, replace=replace).tolist()


def unified_pep(pos_df: pd.DataFrame, num: int) -> pd.DataFrame:
    peps = pos_df["MT_pep"].values
    tcrs = pos_df["CDR3"].values
    hlas = pos_df["HLA_type"].values
    hla_map = build_hla_map(pos_df)

    ts = set(tcrs)
    hs = set(hlas)

    p2t_pos = defaultdict(set)
    p2h_pos = defaultdict(set)
    for pep, tcr, hla in zip(peps, tcrs, hlas):
        p2t_pos[pep].add(tcr)
        p2h_pos[pep].add(hla)

    p2t_neg = {pep: list(ts - p2t_pos[pep]) for pep in p2t_pos}
    p2h_neg = {pep: list(hs - p2h_pos[pep]) for pep in p2h_pos}

    neg_rows = []
    first_half = num // 2
    second_half = num - first_half

    for pep in peps:
        neg_tcrs = _sample_items(p2t_neg[pep], first_half)
        neg_hlas = _sample_items(p2h_neg[pep], first_half)
        for neg_tcr, neg_hla in zip(neg_tcrs, neg_hlas):
            hla_alpha, hla_beta = hla_map[neg_hla]
            neg_rows.append({
                "CDR3": neg_tcr,
                "HLA_type": neg_hla,
                "MT_pep": pep,
                "HLA_alpha": hla_alpha,
                "HLA_beta": hla_beta,
                "label": 0,
                "peplen": len(str(pep)),
            })

    for pep, hla in zip(peps, hlas):
        neg_tcrs = _sample_items(p2t_neg[pep], second_half)
        hla_alpha, hla_beta = hla_map[hla]
        for neg_tcr in neg_tcrs:
            neg_rows.append({
                "CDR3": neg_tcr,
                "HLA_type": hla,
                "MT_pep": pep,
                "HLA_alpha": hla_alpha,
                "HLA_beta": hla_beta,
                "label": 0,
                "peplen": len(str(pep)),
            })

    neg_df = pd.DataFrame(neg_rows)
    neg_df = neg_df.drop_duplicates(subset=["CDR3", "MT_pep", "HLA_type"]).reset_index(drop=True)
    return neg_df


def preprocess(pos_df: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    final_df = pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)
    final_df = final_df.drop_duplicates(subset=["CDR3", "MT_pep", "HLA_type"]).reset_index(drop=True)
    final_df = filter_invalid_sequences(final_df)
    final_df["peplen"] = final_df["MT_pep"].astype(str).str.len()
    ordered_cols = ["CDR3", "HLA_type", "MT_pep", "HLA_alpha", "HLA_beta", "label", "peplen"]
    return final_df[ordered_cols].reset_index(drop=True)


def drop_duplicate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_mix = train_df["MT_pep"].astype(str) + " " + train_df["HLA_type"].astype(str)
    test_mix = test_df["MT_pep"].astype(str) + " " + test_df["HLA_type"].astype(str)
    return test_df.loc[~test_mix.isin(train_mix)].reset_index(drop=True)


def generate_dataset(train_pos_path: str, test_pos_path: str, out_dir: str, sampling: str, neg_ratio: int, seed: int) -> None:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    train_pos = standardize_positive_df(train_pos_path)
    test_pos = standardize_positive_df(test_pos_path)

    if sampling == "shuffle":
        train_neg = random_shuffle(train_pos, neg_ratio)
        test_neg = random_shuffle(test_pos, neg_ratio)
    elif sampling == "unipep":
        train_neg = unified_pep(train_pos, neg_ratio)
        test_neg = unified_pep(test_pos, neg_ratio)
    else:
        raise ValueError("sampling only supports 'shuffle' or 'unipep'")

    train_val_df = preprocess(train_pos, train_neg)
    train_val_df = shuffle(train_val_df, random_state=seed).reset_index(drop=True)

    train_df = train_val_df.sample(frac=0.8, random_state=seed, axis=0)
    val_df = train_val_df.loc[~train_val_df.index.isin(train_df.index)].reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    test_all_df = preprocess(test_pos, test_neg)
    test_df = drop_duplicate(train_val_df, test_all_df)

    train_neg.to_csv(os.path.join(out_dir, "neg_training_data.csv"), index=False)
    test_neg.to_csv(os.path.join(out_dir, "neg_test_data.csv"), index=False)
    train_val_df.to_csv(os.path.join(out_dir, "train_val.csv"), index=False)
    test_all_df.to_csv(os.path.join(out_dir, "test_data_all.csv"), index=False)
    train_df.to_csv(os.path.join(out_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_data.csv"), index=False)

    print(f"[OK] sampling={sampling}")
    print(f"[OK] train_pos={len(train_pos)}, train_neg={len(train_neg)}, train_val={len(train_val_df)}")
    print(f"[OK] train={len(train_df)}, val={len(val_df)}")
    print(f"[OK] test_pos={len(test_pos)}, test_neg={len(test_neg)}, test_all={len(test_all_df)}, test={len(test_df)}")
    print(f"[OK] output_dir={out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MHC II train/val/test datasets with random or unipep negative sampling.")
    parser.add_argument("--train_pos", type=str, default="../data/hlaii_raw_data/pos_training_data.csv", help="Path to the positive training CSV")
    parser.add_argument("--test_pos", type=str, default="../data/hlaii_raw_data/pos_test_data.csv", help="Path to the positive test CSV")
    parser.add_argument("--out_dir", type=str, default="../data/hlaii", help="Output directory")
    parser.add_argument("--sampling", type=str, default="shuffle", help="Negative-sampling method: shuffle or unipep")
    parser.add_argument("--neg_ratio", type=int, default=10, help="Number of negative samples per positive sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        train_pos_path=args.train_pos,
        test_pos_path=args.test_pos,
        out_dir=args.out_dir,
        sampling=args.sampling,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
