import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class GINConv(MessagePassing):
    def __init__(self, emb_dim,aggr='add',eps=0.0,train_eps=False):
        super(GINConv, self).__init__(aggr=aggr)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.tensor([eps], dtype=torch.float))
        else:
            self.register_buffer('eps', torch.tensor([eps], dtype=torch.float))
    def forward(self,x,edge_index):
        aggr_out=self.propagate(edge_index,x=x)
        out=self.mlp((1+self.eps) * x+aggr_out)
        return out
    def message(self, x_j):
        return x_j
    def update(self,aggr_out):
        return aggr_out

class GINEncoderLayer_Pretrain(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers=5,dropout=0.0,JK="last",train_eps=False):
        super(GINEncoderLayer_Pretrain,self).__init__()
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim
        self.JK=JK
        self.dropout_rate=dropout

        self.input_proj=nn.Linear(input_dim,hidden_dim)
        self.type_embedding=nn.Embedding(4,input_dim)

        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.input_proj.weight.data)
        if self.input_proj.bias is not None:
            torch.nn.init.zeros_(self.input_proj.bias.data)

        self.convs=nn.ModuleList()
        self.batch_norms=nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINConv(hidden_dim,aggr="add",eps=0.0,train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    def forward(self,x,edge_index,node_type):

        type_emb = self.type_embedding(node_type)  # [N, input_dim]
        x = x + type_emb
        x=self.input_proj(x)#[M,input_dim]->[M,hidden_dim]
        h_list = [x]
        for layer in range(self.num_layers):
            h=self.convs[layer](h_list[layer],edge_index)
            h=self.batch_norms[layer](h)

            if layer==self.num_layers-1:
                h=F.dropout(h,p=self.dropout_rate,training=self.training)
            else:
                h=F.dropout(F.relu(h),p=self.dropout_rate,training=self.training)
            h_list.append(h)
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_stack = torch.stack(h_list, dim=0)  # [num_layers+1, N, emb_dim]
            node_representation = torch.max(h_stack, dim=0)[0]
        elif self.JK == "sum":
            h_stack = torch.stack(h_list, dim=0)
            node_representation = torch.sum(h_stack, dim=0)
        else:
            raise ValueError("Invalid JK mode.")
        return node_representation

class Pretrain_model(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layer,dropout,JK,train_eps):
        super(Pretrain_model,self).__init__()
        self.encoder=GINEncoderLayer_Pretrain(input_dim,hidden_dim,num_layer,dropout,JK,train_eps)
        self.graph_mlp_128 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        self.graph_mlp_20 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20)
        )
        self.graph_mlp_30=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 30)
        )
        self.projection_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 300 -> 300
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # 300 -> 128
        )

    def forward(self,x,edge_index,node_type,fragment_relation):
        total_node_feature=self.encoder(x,edge_index,node_type)

        fragment_node_mask=(node_type==2)
        graph_node_mask = (node_type == 3)
        fragment_node = total_node_feature[fragment_node_mask]  # [num_fragment_nodes, hidden_dim]
        graph_node = total_node_feature[graph_node_mask]  # [batch_size, hidden_dim]

        graph_task1=self.graph_mlp_128(graph_node)

        graph_task2=self.graph_mlp_20(graph_node)

        fragment_task=self.graph_mlp_30(fragment_node)

        fragment_atoms_list=[]
        fragment_bonds_list=[]

        for atom_ids,bond_ids in fragment_relation:
            atom_feat=total_node_feature[atom_ids]
            bond_feat=total_node_feature[bond_ids]

            atom_mean=atom_feat.mean(dim=0)
            bond_mean=bond_feat.mean(dim=0)

            fragment_atoms_list.append(atom_mean)
            fragment_bonds_list.append(bond_mean)
        fragment_atoms=torch.stack(fragment_atoms_list,dim=0)
        fragment_bonds=torch.stack(fragment_bonds_list,dim=0)
        atom_proj = self.projection_head(fragment_atoms)  # [num_valid_fragments, 128]
        bond_proj = self.projection_head(fragment_bonds)  # [num_valid_fragments, 128]
        return graph_task1,graph_task2,fragment_task,atom_proj,bond_proj

