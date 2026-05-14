import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_topo_loss(graph_task1, topo_fp):
    """
    graph_task1: [batch_size, 128]
    topo_fp:     [batch_size, 128]
    """
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(graph_task1, topo_fp.float())
    return loss

def graph_scaffold_loss(graph_task2, scaffold_props):
    """
    graph_task2:    [batch_size, 20], logits
    scaffold_props: [batch_size, 5]
    """

    ring_logits = graph_task2[:, :10]        # [B, 10]
    aromatic_logits = graph_task2[:, 10:17]  # [B, 7]
    binary_logits = graph_task2[:, 17:]      # [B, 3]

    ring_labels = scaffold_props[:, 0].long()         # [B]
    aromatic_labels = scaffold_props[:, 1].long()     # [B]
    binary_labels = scaffold_props[:, 2:5].float()    # [B, 3]

    loss_ring = F.cross_entropy(ring_logits, ring_labels)
    loss_aromatic = F.cross_entropy(aromatic_logits, aromatic_labels)
    loss_binary = F.binary_cross_entropy_with_logits(binary_logits, binary_labels)

    loss = loss_ring + loss_aromatic + loss_binary
    return loss

def fragment_fg_loss(fragment_task, fragment_fg_labels, fragment_valid_mask):
    """
    fragment_task:       [num_fragment_nodes, 30]
    fragment_fg_labels:  [num_fragment_nodes, 30]
    fragment_valid_mask: [num_fragment_nodes]
    """
    criterion = nn.BCEWithLogitsLoss()
    if fragment_valid_mask.any():
        valid_logits = fragment_task[fragment_valid_mask]
        valid_labels = fragment_fg_labels[fragment_valid_mask].float()
        loss = criterion(valid_logits, valid_labels)
    else:
        loss = fragment_task.sum() * 0.0

    return loss

def atom_bond_contrastive_loss(atom_proj, bond_proj, temperature=0.1):
    """
    atom_proj: [num_valid_fragments, dim]
    bond_proj: [num_valid_fragments, dim]
    """
    # L2 normalize
    atom_proj = F.normalize(atom_proj, dim=1)
    bond_proj = F.normalize(bond_proj, dim=1)

    logits = torch.matmul(atom_proj, bond_proj.t()) / temperature


    labels = torch.arange(logits.size(0), device=logits.device)

    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.t(), labels)

    loss = (loss_a2b + loss_b2a) / 2.0
    return loss