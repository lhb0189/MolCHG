import json
from copy import deepcopy
import rdkit
import rdkit.Chem as Chem
def smi2mol(smiles: str, kekulize=False, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol
def mol2smi(mol):
    return Chem.MolToSmiles(mol)
def get_submol(mol, atom_indices, kekulize=False):

    atom_indices = sorted(set(atom_indices))

    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        if atom_symbol == 'Si':
            atom_symbol = '[Si]'
        if atom_symbol == 'H':
            atom_symbol = '[H]'
        return smi2mol(atom_symbol, kekulize=kekulize)

    atom_set = set(atom_indices)
    edge_indices = []

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in atom_set and a2 in atom_set:
            edge_indices.append(bond.GetIdx())

    submol = Chem.PathToSubmol(mol, edge_indices)
    return submol


class MolInSubgraph:


    def __init__(self, mol, kekulize=False):
        self.mol = mol
        self.kekulize = kekulize
        self.n_atoms = mol.GetNumAtoms()


        self.subgraphs = {i: [i] for i in range(self.n_atoms)}


        self.nei_subgraphs = {i: set() for i in range(self.n_atoms)}

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            self.nei_subgraphs[a1].add(a2)
            self.nei_subgraphs[a2].add(a1)

        self.upid_cnt = self.n_atoms

    def get_nei_smis(self):

        candidates = []
        visited = set()

        for i in self.subgraphs:
            for j in self.nei_subgraphs[i]:
                if j not in self.subgraphs:
                    continue
                if i == j:
                    continue

                pair = tuple(sorted((i, j)))
                if pair in visited:
                    continue
                visited.add(pair)

                merged_atoms = sorted(set(self.subgraphs[i]) | set(self.subgraphs[j]))
                submol = get_submol(self.mol, merged_atoms, kekulize=self.kekulize)
                if submol is None:
                    continue

                smi = mol2smi(submol)
                candidates.append((smi, i, j, merged_atoms))

        return candidates

    def merge(self, i, j):

        if i not in self.subgraphs or j not in self.subgraphs:
            return None

        new_id = self.upid_cnt
        self.upid_cnt += 1

        new_atoms = sorted(set(self.subgraphs[i]) | set(self.subgraphs[j]))


        new_neighbors = (self.nei_subgraphs[i] | self.nei_subgraphs[j]) - {i, j}


        self.subgraphs[new_id] = new_atoms
        self.nei_subgraphs[new_id] = set()


        for nb in new_neighbors:
            if nb in self.nei_subgraphs:
                self.nei_subgraphs[nb].discard(i)
                self.nei_subgraphs[nb].discard(j)
                self.nei_subgraphs[nb].add(new_id)
                self.nei_subgraphs[new_id].add(nb)


        del self.subgraphs[i]
        del self.subgraphs[j]
        del self.nei_subgraphs[i]
        del self.nei_subgraphs[j]

        return new_id


class Tokenizer:


    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.kekulize = False
        self.vocab_dict = self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path):

        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            raise ValueError("vocab.txt is empty")

        vocab_dict = {}

        start_idx = 0
        try:
            config = json.loads(lines[0])
            if isinstance(config, dict):
                self.kekulize = bool(config.get("kekulize", False))
                start_idx = 1
        except Exception:
            start_idx = 0

        for line in lines[start_idx:]:
            parts = line.split("\t")

            if len(parts) >= 3:
                smi = parts[0]
                try:
                    freq = int(parts[2])
                except Exception:
                    try:
                        freq = float(parts[2])
                    except Exception:
                        freq = 1
            elif len(parts) == 2:
                smi = parts[0]
                try:
                    freq = int(parts[1])
                except Exception:
                    try:
                        freq = float(parts[1])
                    except Exception:
                        freq = 1
            else:
                smi = parts[0]
                freq = 1

            vocab_dict[smi] = freq

        return vocab_dict

    def tokenize(self, mol):

        mol_subgraph = MolInSubgraph(mol, kekulize=self.kekulize)

        while True:
            candidates = mol_subgraph.get_nei_smis()

            valid_candidates = []
            for smi, i, j, merged_atoms in candidates:
                if smi in self.vocab_dict:
                    valid_candidates.append((self.vocab_dict[smi], smi, i, j, merged_atoms))

            if len(valid_candidates) == 0:
                break


            valid_candidates.sort(key=lambda x: (-x[0], len(x[4]), x[2], x[3]))
            _, _, best_i, best_j, _ = valid_candidates[0]

            mol_subgraph.merge(best_i, best_j)


        final_ids = sorted(mol_subgraph.subgraphs.keys())
        cliques = [sorted(mol_subgraph.subgraphs[fid]) for fid in final_ids]

        fid2cid = {fid: idx for idx, fid in enumerate(final_ids)}

        edges = set()
        for fid in final_ids:
            for nb in mol_subgraph.nei_subgraphs[fid]:
                if nb in fid2cid:
                    c1 = fid2cid[fid]
                    c2 = fid2cid[nb]
                    if c1 != c2:
                        edges.add(tuple(sorted((c1, c2))))

        edges = [list(e) for e in sorted(edges)]
        return cliques, edges
def principal_subgraph_decomp(mol, vocab_path="vocab.txt"):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return [], []
    if n_atoms == 1:
        return [[0]], []
    tokenizer = Tokenizer(vocab_path)
    cliques, edges = tokenizer.tokenize(mol)
    return cliques, edges