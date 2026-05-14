import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from graph import MolGraph
from model import GINEncoderLayer_Pretrain

def finetune_preprocess_and_save(csv_path, save_path, label_cols, task_type="classification"):

    df = pd.read_csv(csv_path)
    smiles_list = df["smiles"].tolist()

    if isinstance(label_cols, str):
        label_cols = [label_cols]
    labels_df = df[label_cols]

    data_list = []
    failed = []

    for i, smi in enumerate(tqdm(smiles_list, desc="Preprocessing for finetune")):
        try:
            mol_graph = MolGraph(smi)

            label_values = labels_df.iloc[i].values.astype(float)
            label_tensor = torch.tensor(label_values, dtype=torch.float32)
            label_tensor = torch.where(
                torch.isnan(label_tensor),
                torch.tensor(-1.0),
                label_tensor
            )

            data = {
                "smiles": smi,
                "x": mol_graph.x,
                "edge_index": mol_graph.edge_index,
                "node_type": mol_graph.node_type,
                "num_part": mol_graph.num_part,
                "labels": label_tensor,
            }
            data_list.append(data)

        except Exception as e:
            failed.append((i, smi, str(e)))

    torch.save({"data_list": data_list, "failed": failed}, save_path)
    print(f"Saved to {save_path}")
    print(f"Valid: {len(data_list)}, Failed: {len(failed)}, Labels: {label_cols}")


class FinetuneDataset(Dataset):
    """微调专用 Dataset"""

    def __init__(self, pt_path):
        obj = torch.load(pt_path, weights_only=False)
        self.data_list = obj["data_list"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def finetune_collate_fn(batch):
    x_list = []
    edge_index_list = []
    node_type_list = []
    mol_scope_list = []
    labels_list = []
    node_offset = 0
    for item in batch:
        x = item["x"]
        edge_index = item["edge_index"]
        node_type = item["node_type"]
        labels = item["labels"]

        num_nodes = x.size(0)

        x_list.append(x)
        edge_index_list.append(edge_index + node_offset)
        node_type_list.append(node_type)
        mol_scope_list.append([node_offset, num_nodes])
        labels_list.append(labels)

        node_offset += num_nodes

    batch_dict = {
        "x": torch.cat(x_list, dim=0),  # [total_nodes, feature_dim]
        "edge_index": torch.cat(edge_index_list, dim=1),  # [2, total_edges]
        "node_type": torch.cat(node_type_list, dim=0),  # [total_nodes]
        "mol_scope": torch.tensor(mol_scope_list, dtype=torch.long).t().contiguous(),  # [2, B]
        "labels": torch.stack(labels_list, dim=0),  # [B, num_tasks]
    }
    return batch_dict
class FinetuneModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, JK, train_eps, num_tasks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = GINEncoderLayer_Pretrain(
            input_dim, hidden_dim, num_layers, dropout, JK, train_eps
        )
        self.pred_head = nn.Linear(hidden_dim * 4, num_tasks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, node_type, mol_scope):
        """
        x:          [total_nodes, feature_dim]
        edge_index: [2, total_edges]
        node_type:  [total_nodes]  (0=atom, 1=bond, 2=fragment, 3=graph)
        mol_scope:  [2, batch_size]
        """
        node_repr = self.encoder(x, edge_index, node_type)

        batch_size = mol_scope.size(1)
        device = x.device

        batch_index = torch.zeros(x.size(0), dtype=torch.long, device=device)
        for i in range(batch_size):
            start = mol_scope[0, i].item()
            num = mol_scope[1, i].item()
            batch_index[start: start + num] = i


        graph_repr = node_repr[node_type == 3]  # [B, hidden_dim]

        atom_pool = self._scatter_mean(node_repr, node_type, 0, batch_index, batch_size)
        bond_pool = self._scatter_mean(node_repr, node_type, 1, batch_index, batch_size)
        frag_pool = self._scatter_mean(node_repr, node_type, 2, batch_index, batch_size)


        combined = torch.cat([atom_pool, bond_pool, frag_pool, graph_repr], dim=1)
        # combined: [B, 4 * hidden_dim]

        out = self.pred_head(self.dropout(combined))
        return out

    def _scatter_mean(self, node_repr, node_type, type_id, batch_index, batch_size):
        device = node_repr.device
        hidden_dim = node_repr.size(1)
        mask = (node_type == type_id)

        if not mask.any():
            return torch.zeros(batch_size, hidden_dim, device=device)

        typed_repr = node_repr[mask]  # [num_typed, hidden_dim]
        typed_batch = batch_index[mask]  # [num_typed]

        # scatter add + count
        out = torch.zeros(batch_size, hidden_dim, device=device)
        count = torch.zeros(batch_size, 1, device=device)
        out.scatter_add_(0, typed_batch.unsqueeze(1).expand_as(typed_repr), typed_repr)
        count.scatter_add_(0, typed_batch.unsqueeze(1),
                           torch.ones(typed_batch.size(0), 1, device=device))
        count = count.clamp(min=1)
        return out / count


def load_pretrained_encoder(model, pretrain_path):
    state_dict = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    encoder_state_dict = {
        k.replace("encoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state_dict)
    print(f"Loaded encoder weights from {pretrain_path}")
    return model