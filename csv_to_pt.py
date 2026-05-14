import os
import torch
import pandas as pd
from tqdm import tqdm
from graph import MolGraph


LABEL_COLS_MAP = {
    "bbbp":          ["p_np"],
    "bace":          ["Class"],
    "clintox":       ["CT_TOX", "FDA_APPROVED"],
    "hiv":           ["HIV_active"],
    "tox21":         ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
                      "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
                      "SR-HSE", "SR-MMP", "SR-p53"],
    "sider":         ["Hepatobiliary disorders", "Metabolism and nutrition disorders",
                      "Product issues", "Eye disorders", "Investigations",
                      "Musculoskeletal and connective tissue disorders",
                      "Gastrointestinal disorders", "Social circumstances",
                      "Immune system disorders", "Reproductive system and breast disorders",
                      "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                      "General disorders and administration site conditions",
                      "Endocrine disorders", "Surgical and medical procedures",
                      "Vascular disorders", "Blood and lymphatic system disorders",
                      "Skin and subcutaneous tissue disorders",
                      "Congenital, familial and genetic disorders",
                      "Infections and infestations",
                      "Respiratory, thoracic and mediastinal disorders",
                      "Psychiatric disorders", "Renal and urinary disorders",
                      "Pregnancy, puerperium and perinatal conditions",
                      "Ear and labyrinth disorders", "Cardiac disorders",
                      "Nervous system disorders",
                      "Injury, poisoning and procedural complications"],
    "esol":          ["measured log solubility in mols per litre"],
    "freesolv":      ["expt"],
    "lipophilicity": ["exp"],
    "toxcast":       "auto",
}


def detect_smiles_col(df):
    for name in ["smiles", "SMILES", "Smiles", "mol", "Mol"]:
        if name in df.columns:
            return name
    raise ValueError(f"can not find: {list(df.columns)}")


def detect_label_cols(df, csv_name):

    key = csv_name.lower()
    if key in LABEL_COLS_MAP:
        label_cols = LABEL_COLS_MAP[key]
        if label_cols == "auto":
            smiles_col = detect_smiles_col(df)
            label_cols = [c for c in df.columns if c != smiles_col]
        return label_cols

    smiles_col = detect_smiles_col(df)
    label_cols = [c for c in df.columns if c != smiles_col]
    print(f"[WARN]  '{csv_name} can not find',  {len(label_cols)} labels")
    return label_cols


def convert(csv_path, output_path):
    df = pd.read_csv(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    smiles_col = detect_smiles_col(df)
    label_cols = detect_label_cols(df, csv_name)

    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV lack: {missing}")

    print(f"CSV:        {csv_path} ({len(df)} rows)")
    print(f"SMILES col: {smiles_col}")
    print(f"Label cols: {label_cols} ({len(label_cols)} tasks)")

    smiles_list = df[smiles_col].tolist()
    labels_df = df[label_cols]

    data_list = []
    failed = []

    for i, smi in enumerate(tqdm(smiles_list, desc="Converting")):
        try:
            mol_graph = MolGraph(smi)

            label_values = labels_df.iloc[i].values.astype(float)
            label_tensor = torch.tensor(label_values, dtype=torch.float32)
            label_tensor = torch.where(
                torch.isnan(label_tensor),
                torch.tensor(-1.0),
                label_tensor
            )

            data_list.append({
                "smiles": smi,
                "x": mol_graph.x,
                "edge_index": mol_graph.edge_index,
                "node_type": mol_graph.node_type,
                "num_part": mol_graph.num_part,
                "labels": label_tensor,
            })
        except Exception as e:
            failed.append((i, smi, str(e)))

    torch.save({"data_list": data_list, "failed": failed}, output_path)

    print(f"\nDone!")
    print(f"  Valid:  {len(data_list)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Saved:  {output_path}")


if __name__ == "__main__":
    csv_path = "Finetune_Datasets\\csv_file\\ESOL.csv"
    name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = f"{name}_molgraph.pt"
    convert(csv_path, output_path)
