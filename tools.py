from graph import MolGraph
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
import torch
import logging

def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'debug.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    return log


def preprocess_and_save(csv_path, save_path):
    df = pd.read_csv(csv_path)
    smiles_list = df["smiles"].tolist()
    data_list = []
    failed = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Preprocessing MolGraph")):
        if i % 100 == 0:
            print(f"[Progress] Processed {i} samples")
        try:
            mol_graph = MolGraph(smi)

            data = {
                "smiles": smi,
                "x": mol_graph.x,
                "edge_index": mol_graph.edge_index,
                "node_type": mol_graph.node_type,
                "num_part": mol_graph.num_part,
                "topo_fp": mol_graph.topo_fp,
                "scaffold_props": mol_graph.scaffold_props,
                "fragment_fg_labels": mol_graph.fragment_fg_labels,
                "fragment_valid_mask": mol_graph.fragment_valid_mask,
                "fragment_relation": [
                    (
                        torch.tensor(atom_ids, dtype=torch.long),
                        torch.tensor(bond_ids, dtype=torch.long)
                    )
                    for atom_ids, bond_ids in mol_graph.fragment_relation
                ],
            }
            data_list.append(data)

        except Exception as e:
            failed.append((i, smi, str(e)))
    torch.save({
        "data_list": data_list,
        "failed": failed
    }, save_path)
    print(f"Saved to {save_path}")
    print(f"Valid samples: {len(data_list)}")
    print(f"Failed samples: {len(failed)}")

class MolGraphDataset(Dataset):
    def __init__(self, pt_path):
        obj=torch.load(pt_path,weights_only=False)
        self.data_list=obj["data_list"]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def molgraph_collate_fn(batch):
    x_list = []
    edge_index_list = []
    node_type_list = []
    num_part_list = []
    mol_scope_list = []
    topo_fp_list = []
    scaffold_props_list = []
    fragment_fg_labels_list = []
    fragment_relation_list = []
    fragment_valid_mask_list=[]
    node_offset = 0
    for item in batch:
        x = item["x"]
        edge_index = item["edge_index"]
        node_type = item["node_type"]
        num_part = item["num_part"]
        topo_fp = item["topo_fp"]
        scaffold_props = item["scaffold_props"]
        fragment_fg_labels = item["fragment_fg_labels"]
        fragment_valid_mask = item["fragment_valid_mask"]
        fragment_relation = item["fragment_relation"]

        num_nodes = x.size(0)

        x_list.append(x)
        edge_index_list.append(edge_index + node_offset)
        node_type_list.append(node_type)
        num_part_list.append(num_part)
        mol_scope_list.append([node_offset, num_nodes])
        topo_fp_list.append(topo_fp)
        scaffold_props_list.append(scaffold_props)
        fragment_valid_mask_list.append(fragment_valid_mask)
        fragment_fg_labels_list.append(fragment_fg_labels)

        for atom_ids, bond_ids in fragment_relation:
            fragment_relation_list.append((
                atom_ids + node_offset,
                bond_ids + node_offset
            ))

        node_offset += num_nodes
    batch_dict = {
        "x": torch.cat(x_list, dim=0),#[total_node,feature_dim]
        "edge_index": torch.cat(edge_index_list, dim=1),#[total_edge,2]
        "node_type": torch.cat(node_type_list, dim=0),#[total_node,1]
        "mol_scope": torch.tensor(mol_scope_list, dtype=torch.long).t().contiguous(),#[batch_size,2]
        "num_part": torch.stack(num_part_list, dim=0),#[batch_size,3]
        "topo_fp": torch.stack(topo_fp_list, dim=0),#[Batch_size,128]
        "scaffold_props": torch.stack(scaffold_props_list, dim=0),#[batch_size,5],
        "fragment_fg_labels": torch.cat(fragment_fg_labels_list, dim=0), # [total_fragment_nodes, 30]
        "fragment_valid_mask": torch.cat(fragment_valid_mask_list, dim=0),#[total_fragment_nodes]
        "fragment_relation": fragment_relation_list # list of (atom_idx_tensor, bond_idx_tensor)
    }
    return batch_dict
