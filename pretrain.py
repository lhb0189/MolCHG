import torch
from graph import MolGraph
from torch.utils.data import Dataset,DataLoader,random_split
from model import Pretrain_model
from tools import molgraph_collate_fn,MolGraphDataset,set_log
from loss_function import graph_topo_loss,graph_scaffold_loss,fragment_fg_loss,atom_bond_contrastive_loss
log_save_path="pretrain_logs"
log=set_log("pretrain",log_save_path)
total_dataset=MolGraphDataset("pretrain_molgraph.pt")
total_size=len(total_dataset)
train_size=int(0.95*total_size)
val_size = total_size - train_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset,val_dataset=random_split(total_dataset,[train_size,val_size],generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    collate_fn=molgraph_collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=molgraph_collate_fn,
    num_workers=0
)

epochs=100
input_dim=15
hidden_dim=300
num_layers=5
dropout=0.5
JK="sum"
train_eps=True

model=Pretrain_model(input_dim,hidden_dim,num_layers,dropout,JK,train_eps).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_total_loss=0.0
    train_topo_loss=0.0
    train_scaffold_loss=0.0
    train_fragment_loss=0.0
    train_contrast_loss=0.0
    for batch in train_loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        node_type = batch["node_type"].to(device)
        topo_fp=batch["topo_fp"].to(device)
        scaffold_props=batch["scaffold_props"].to(device)
        fragment_fg_labels=batch["fragment_fg_labels"].to(device)
        fragment_valid_mask = batch["fragment_valid_mask"].to(device)
        fragment_relation = [
            (atom_ids.to(device), bond_ids.to(device))
            for atom_ids, bond_ids in batch["fragment_relation"]
        ]

        optimizer.zero_grad()

        graph_task1,graph_task2,fragment_task,atom_proj,bond_proj=model(x,edge_index,node_type,fragment_relation)
        loss_topo = graph_topo_loss(graph_task1, topo_fp) #graph_node 预测Topological fingperirnts
        loss_scaffold = graph_scaffold_loss(graph_task2, scaffold_props) #graph_node 预测骨架性质
        loss_fragment = fragment_fg_loss(
            fragment_task,
            fragment_fg_labels,
            fragment_valid_mask
        )
        loss_contrast = atom_bond_contrastive_loss(atom_proj, bond_proj, temperature=0.1) #Atom_node和bond_node基于Fragment去做对比学习

        total_loss=0.4*loss_topo+0.4*loss_scaffold+0.4*loss_fragment+0.2*loss_contrast

        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item()
        train_topo_loss += loss_topo.item()
        train_scaffold_loss += loss_scaffold.item()
        train_fragment_loss += loss_fragment.item()
        train_contrast_loss += loss_contrast.item()

    train_total_loss /= len(train_loader)
    train_topo_loss /= len(train_loader)
    train_scaffold_loss /= len(train_loader)
    train_fragment_loss /= len(train_loader)
    train_contrast_loss /= len(train_loader)

    model.eval()

    val_total_loss = 0.0
    val_topo_loss = 0.0
    val_scaffold_loss = 0.0
    val_fragment_loss = 0.0
    val_contrast_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            node_type = batch["node_type"].to(device)
            topo_fp = batch["topo_fp"].to(device)
            scaffold_props = batch["scaffold_props"].to(device)
            fragment_fg_labels = batch["fragment_fg_labels"].to(device)
            fragment_valid_mask = batch["fragment_valid_mask"].to(device)
            fragment_relation = [
                (atom_ids.to(device), bond_ids.to(device))
                for atom_ids, bond_ids in batch["fragment_relation"]
            ]

            graph_task1, graph_task2, fragment_task, atom_proj, bond_proj = model(
                x, edge_index, node_type, fragment_relation
            )

            loss_topo = graph_topo_loss(graph_task1, topo_fp)
            loss_scaffold = graph_scaffold_loss(graph_task2, scaffold_props)
            loss_fragment = fragment_fg_loss(fragment_task,fragment_fg_labels, fragment_valid_mask)
            loss_contrast = atom_bond_contrastive_loss(atom_proj, bond_proj, temperature=0.1)

            total_loss = (
                0.4*loss_topo +
                0.4*loss_scaffold +
                0.4*loss_fragment +
                0.2*loss_contrast
            )

            val_total_loss += total_loss.item()
            val_topo_loss += loss_topo.item()
            val_scaffold_loss += loss_scaffold.item()
            val_fragment_loss += loss_fragment.item()
            val_contrast_loss += loss_contrast.item()
    val_total_loss /= len(val_loader)
    val_topo_loss /= len(val_loader)
    val_scaffold_loss /= len(val_loader)
    val_fragment_loss /= len(val_loader)
    val_contrast_loss /= len(val_loader)

    log.info(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Total: {train_total_loss:.4f} | "
        f"Topo: {train_topo_loss:.4f} | "
        f"Scaffold: {train_scaffold_loss:.4f} | "
        f"Fragment: {train_fragment_loss:.4f} | "
        f"Contrast: {train_contrast_loss:.4f} || "
        f"Val Total: {val_total_loss:.4f} | "
        f"Topo: {val_topo_loss:.4f} | "
        f"Scaffold: {val_scaffold_loss:.4f} | "
        f"Fragment: {val_fragment_loss:.4f} | "
        f"Contrast: {val_contrast_loss:.4f}"
    )

    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        torch.save(model.state_dict(), "best_pretrain_model.pth")
        log.info(f"Best model saved at epoch {epoch + 1}, val_loss={best_val_loss:.4f}")