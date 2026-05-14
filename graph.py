import torch
from rdkit import Chem

from rdkit.Chem import RDKFingerprint
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from Principal_SubGraph_mining import principal_subgraph_decomp

top30_functional_groups = [
    "aromatic_ring",
    "carbonyl",
    "amide",
    "amine_secondary",
    "ether",
    "aniline",
    "amine_tertiary",
    "halogen",
    "thioether",
    "urea",
    "hydroxyl",
    "sulfone",
    "lactam",
    "ester",
    "alcohol_aliphatic",
    "sulfonamide",
    "amine_primary",
    "nitrile",
    "imine",
    "ketone",
    "imide",
    "phenol",
    "hydrazine",
    "hydrazone",
    "acetal",
    "thiocarbonyl",
    "carbamate",
    "sulfoxide",
    "azo",
    "aldehyde"
]

functional_group_smarts = {
    "aromatic_ring": "[a;r]",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine_secondary": "[NX3;H1;!$(N[C,S]=O)]([#6])[#6]",
    "ether": "[OD2]([#6])[#6]",
    "aniline": "[NX3;!$(N[C,S]=O)][c]",
    "amine_tertiary": "[NX3;H0;!$(N[C,S]=O)]([#6])([#6])[#6]",
    "halogen": "[F,Cl,Br,I]",
    "thioether": "[SD2]([#6])[#6]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "hydroxyl": "[OX2H]",
    "sulfone": "[SX4](=[OX1])(=[OX1])",
    "lactam": "[NX3;R][CX3;R](=[OX1])",
    "ester": "[#6][CX3](=[OX1])[OX2H0][#6]",
    "alcohol_aliphatic": "[CX4][OX2H]",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])",
    "amine_primary": "[NX3;H2;!$(N[C,S]=O)][#6]",
    "nitrile": "[CX2]#N",
    "imine": "[CX3]=[NX2]",
    "ketone": "[#6][CX3](=[OX1])[#6]",
    "imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "phenol": "[OX2H][c]",
    "hydrazine": "[NX3][NX3]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    "acetal": "[CX4]([OX2][#6])([OX2][#6])",
    "thiocarbonyl": "[CX3]=[SX1]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",
    "sulfoxide": "[SX3](=[OX1])",
    "azo": "[NX2]=[NX2]",
    "aldehyde": "[CX3H1](=O)[#6]"
}

compiled_fg_patterns = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in functional_group_smarts.items()
}

def detect_functional_groups_in_clique(mol, clique, fg_patterns):
    atom_set = set(clique)
    fg_dict = {}
    for fg_name, patt in fg_patterns.items():
        if patt is None:
            fg_dict[fg_name] = 0
            continue
        matches = mol.GetSubstructMatches(patt)
        found = 0
        for match in matches:
            if set(match).issubset(atom_set):
                found = 1
                break
        fg_dict[fg_name] = found
    return fg_dict
def build_fragment_fg_labels(mol, cliques, fg_patterns, fg_names):
    all_labels = []
    for clique in cliques:
        fg_dict = detect_functional_groups_in_clique(mol, clique, fg_patterns)
        label = [float(fg_dict[name]) for name in fg_names]
        all_labels.append(label)
    if len(all_labels) == 0:
        return torch.zeros((0, len(fg_names)), dtype=torch.float32)
    return torch.tensor(all_labels, dtype=torch.float32)

def get_topological_fingerprint_and_scaffold_property(smiles,fp_size):
    mol=Chem.MolFromSmiles(smiles)
    fp = RDKFingerprint(mol, maxPath=7, fpSize=fp_size)
    arr = np.zeros((fp_size,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    topo_fp=torch.tensor(arr,dtype=torch.float32)
    num_rings=mol.GetRingInfo().NumRings()
    aromatic_rings = 0
    ring_info = mol.GetRingInfo()
    for ring in ring_info.BondRings():
        if all([mol.GetBondWithIdx(b).GetIsAromatic() for b in ring]):
            aromatic_rings += 1
    fused = 0
    atom_rings = [set(r) for r in ring_info.AtomRings()]
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if len(atom_rings[i] & atom_rings[j]) >= 2:
                fused = 1
                break
        if fused:
            break

    hetero = 0
    for ring in atom_rings:
        for idx in ring:
            if mol.GetAtomWithIdx(idx).GetAtomicNum() not in [6,1]:
                hetero = 1
                break
        if hetero:
            break

    bridged = 1 if rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0 else 0
    scaffold_props = torch.tensor(
        [num_rings, aromatic_rings, fused, hetero, bridged],
        dtype=torch.float32,
    )
    return topo_fp, scaffold_props
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"{x} not in allowable_set")
    return [1 if x == s else 0 for s in allowable_set]


def get_atom_features(atom,mol):
    atomic_num = float(atom.GetAtomicNum())
    degree = float(atom.GetDegree())
    formal_charge = float(atom.GetFormalCharge())
    num_radical_electrons = float(atom.GetNumRadicalElectrons())
    hyb = atom.GetHybridization()
    hybridization_feat = [
        1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3D else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3D2 else 0.0,
        1.0 if hyb not in [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ] else 0.0,
    ]
    mass = float(atom.GetMass() * 0.01)
    total_num_hs = float(atom.GetTotalNumHs())
    is_chiral_center = 0.0
    chirality_feat = [0.0, 0.0]  # [R, S]
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    atom_idx = atom.GetIdx()
    for idx, label in chiral_centers:
        if idx == atom_idx:
            is_chiral_center = 1.0
            if label == 'R':
                chirality_feat = [1.0, 0.0]
            elif label == 'S':
                chirality_feat = [0.0, 1.0]
            else:
                chirality_feat = [0.0, 0.0]
    attributes=[]
    attributes.append(atomic_num)
    attributes.append(degree)
    attributes.append(formal_charge)
    attributes.append(num_radical_electrons)
    attributes.extend(hybridization_feat)
    attributes.append(mass)
    attributes.append(total_num_hs)
    attributes.append(is_chiral_center)
    attributes.extend(chirality_feat)
    return attributes

bond_fdim=15

def get_bond_features(bond):
    if bond is None:
        fbond=[1]+[0]*(bond_fdim-1)
    else:
        bt=bond.GetBondType()
        bd=bond.GetBondDir()
        fbond= [
                0,
                1 if bt == Chem.rdchem.BondType.SINGLE else 0,
                1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
                1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
                1 if bond.GetBeginAtom().GetAtomicNum() != bond.GetEndAtom().GetAtomicNum() else 0,
                1 if bd == Chem.rdchem.BondDir.ENDUPRIGHT else 0,
                1 if bd == Chem.rdchem.BondDir.ENDDOWNRIGHT else 0,
                abs(bond.GetBeginAtom().GetFormalCharge() - bond.GetEndAtom().GetFormalCharge()),
                1 if bond.GetIsConjugated() else 0,
            ]
        fbond += one_of_k_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond

def get_fragment_feature(mol,fragment_atom_indices):

    if fragment_atom_indices is None or len(fragment_atom_indices) == 0:
        raise ValueError("fragment_atom_indices has no element")
    fragment_set = set(fragment_atom_indices)
    atoms = [mol.GetAtomWithIdx(i) for i in fragment_atom_indices]
    num_atoms=len(atoms)
    num_bonds=0
    num_single_bonds=0
    num_double_bonds=0
    num_triple_bonds=0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in fragment_set and a2 in fragment_set:
            num_bonds+=1
            bond_type=bond.GetBondType()
            if bond_type.name == "SINGLE":
                num_single_bonds += 1
            elif bond_type.name == "DOUBLE":
                num_double_bonds += 1
            elif bond_type.name == "TRIPLE":
                num_triple_bonds += 1
    num_hetero=sum(1 for atom in atoms if atom.GetAtomicNum() not in [1,6])
    avg_atomic_mass = sum(atom.GetMass() for atom in atoms) / num_atoms
    formal_charge_sum=sum(atom.GetFormalCharge() for atom in atoms)
    total_num_hs=sum(atom.GetTotalNumHs() for atom in atoms)
    num_c = sum(1 for atom in atoms if atom.GetAtomicNum() == 6)
    num_n = sum(1 for atom in atoms if atom.GetAtomicNum() == 7)
    num_o = sum(1 for atom in atoms if atom.GetAtomicNum() == 8)
    total_valence_sum=sum(atom.GetTotalValence() for atom in atoms)
    halogen_atomic_nums={9,17,35,53}
    num_halogens = sum(1 for atom in atoms if atom.GetAtomicNum() in halogen_atomic_nums)
    avg_degree=sum(atom.GetDegree() for atom in atoms) / num_atoms
    feature = [float(num_atoms),float(num_bonds),float(num_hetero),float(avg_atomic_mass),float(formal_charge_sum),
        float(total_num_hs),float(num_c),float(num_n),float(num_o),float(num_single_bonds),
        float(num_double_bonds),float(num_triple_bonds),float(total_valence_sum),float(num_halogens),float(avg_degree)
    ]
    return feature

def get_virtual_node_feature(mol,cliques):
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    num_fragments = len(cliques)
    frag_atom_counts=[len(c) for c in cliques]
    frag_bond_counts = []
    for clique in cliques:
        atom_set = set(clique)
        bond_count = 0
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in atom_set and a2 in atom_set:
                bond_count += 1
        frag_bond_counts.append(bond_count)
    f1=float(num_atoms)
    f2=float(num_bonds)
    f3=float(num_fragments)
    f4=float(sum(frag_atom_counts)/num_fragments)
    f5=float(sum(frag_bond_counts)/num_fragments)
    f6=float(max(frag_atom_counts))
    f7=float(max(frag_bond_counts))
    f8=float(min(frag_atom_counts))
    f9=float(min(frag_bond_counts))
    f10=float(sum(1 for x in frag_atom_counts if x<=3))
    f11=float(sum(1 for x in frag_atom_counts if 4<=x<=6))
    f12=float(sum(1 for x in frag_atom_counts if x>=7))
    f13=float(sum(1 for x in frag_atom_counts if x==1))
    atom_mean = sum(frag_atom_counts) / num_fragments
    f14 = float(sum((x - atom_mean) ** 2 for x in frag_atom_counts) / num_fragments)
    bond_mean = sum(frag_bond_counts) / num_fragments
    f15 = float(sum((x - bond_mean) ** 2 for x in frag_bond_counts) / num_fragments)
    feature = [
        f1, f2, f3, f4, f5,
        f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15
    ]
    return feature

class MolGraph(object):

    def __init__(self,smiles):
        self.smiles=smiles
        self.mol=Chem.MolFromSmiles(smiles)
        mol=self.mol
        cliques,frag_edges=principal_subgraph_decomp(mol)
        self.cliques=cliques
        self.frag_edges=frag_edges
        fragment_fg_labels = build_fragment_fg_labels(
            mol=mol,
            cliques=cliques,
            fg_patterns=compiled_fg_patterns,
            fg_names=top30_functional_groups
        )
        self.fragment_fg_labels = fragment_fg_labels
        self.fragment_valid_mask = torch.tensor(
            [len(clique) > 1 for clique in cliques],
            dtype=torch.bool
        )
        N=mol.GetNumAtoms()
        M=mol.GetNumBonds()
        F=len(cliques)
        self.num_atoms=N
        self.num_bonds=M
        self.num_frags=F
        atom_offset=0
        bond_offset=N
        frag_offset=N+M
        graph_idx=N+M+F
        self.atom_offset = atom_offset
        self.bond_offset = bond_offset
        self.frag_offset = frag_offset
        self.graph_idx = graph_idx
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = get_atom_features(atom, mol)
            atom_features_list.append(atom_feature)
        bond_features_list=[]
        for bond in mol.GetBonds():
            bond_features=get_bond_features(bond)
            bond_features_list.append(bond_features)
        fragment_features_list=[]
        for clique in cliques:
            frag_feature=get_fragment_feature(mol,clique)
            fragment_features_list.append(frag_feature)
        graph_feature=get_virtual_node_feature(mol,cliques)
        all_features = (
                atom_features_list +
                bond_features_list +
                fragment_features_list +
                [graph_feature]
        )
        feature_dim=len(all_features[0])
        self.feature_dim=feature_dim
        self.x=torch.tensor(all_features,dtype=torch.float)

        edge_list=[]

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            u = atom_offset + a1
            v = atom_offset + a2
            edge_list.append([u,v])
            edge_list.append([v,u])

        atom_to_bonds={}
        for atom in mol.GetAtoms():
            atom_idx=atom.GetIdx()
            atom_to_bonds[atom_idx]=[bond.GetIdx() for bond in atom.GetBonds()]
        for atom_idx, connected_bonds in atom_to_bonds.items():
            for i in range(len(connected_bonds)):
                for j in range(i+1,len(connected_bonds)):
                    b1=connected_bonds[i]
                    b2=connected_bonds[j]
                    u=bond_offset + b1
                    v=bond_offset + b2
                    edge_list.append([u,v])
                    edge_list.append([v,u])
        for f1,f2 in frag_edges:
            u=frag_offset + f1
            v=frag_offset + f2
            edge_list.append([u, v])
            edge_list.append([v, u])
        self.fragment_relation=[]
        for frag_id, clique in enumerate(cliques):

            if len(clique) == 1:
                continue
            atom_set = set(clique)
            atom_node_indices = [atom_offset + atom_idx for atom_idx in clique]

            bond_node_indices = []
            for bond in mol.GetBonds():
                b_idx = bond.GetIdx()
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_set and a2 in atom_set:
                    bond_node = bond_offset + b_idx
                    bond_node_indices.append(bond_node)

            self.fragment_relation.append([atom_node_indices, bond_node_indices])

        for frag_id, clique in enumerate(cliques):
            frag_node = frag_offset + frag_id
            for atom_idx in clique:
                atom_node = atom_offset + atom_idx
                edge_list.append([frag_node, atom_node])
                edge_list.append([atom_node, frag_node])

        for frag_id, clique in enumerate(cliques):
            frag_node = frag_offset + frag_id
            atom_set = set(clique)
            for bond in mol.GetBonds():
                b_idx = bond.GetIdx()
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_set and a2 in atom_set:
                    bond_node = bond_offset + b_idx
                    edge_list.append([frag_node, bond_node])
                    edge_list.append([bond_node, frag_node])

        for frag_id in range(F):
            frag_node = frag_offset + frag_id
            edge_list.append([graph_idx, frag_node])
            edge_list.append([frag_node, graph_idx])
        self.edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()#获取的属性
        self.num_part=torch.tensor([N,M,F],dtype=torch.long)
        self.node_type = torch.tensor(
            [0] * N + [1] * M + [2] * F + [3],
            dtype=torch.long
        )
        topo_fp,scaffold_props=get_topological_fingerprint_and_scaffold_property(smiles,fp_size=128)
        self.topo_fp=topo_fp
        self.scaffold_props=scaffold_props
    def get_features(self):
        return self.x,self.edge_index,self.num_part,self.node_type,self.topo_fp,self.scaffold_props,self.fragment_fg_labels, self.fragment_valid_mask,self.fragment_relation
