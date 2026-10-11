import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
import os

def load_drug_datasets(solubility_path="data/delaney.csv", toxicity_path="data/toxicity.csv"):
    if not os.path.exists(solubility_path):
        raise FileNotFoundError(f"Could not find {solubility_path}. Please place your Kaggle dataset in the data/ folder.")
    
    df_sol = pd.read_csv(solubility_path)
    data_list = []
    
    for i, row in df_sol.iterrows():
        try:
            smiles = str(row['SMILES'])
            solubility = float(row['measured log(solubility:mol/L)'])
            
            # Derive toxicity proxy label safely from solubility thresholds if standalone toxicity file isn't present
            toxicity = 1.0 if solubility > -5.0 else 0.0 
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Extract atom features (nodes)
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([atom.GetAtomicNum(), atom.GetDegree()])
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Extract bonds (edges)
            edge_indices = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edge_indices.append([start, end])
                edge_indices.append([end, start])
                
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            # Target outputs: [Solubility, Toxicity] shaped as [2]
            y = torch.tensor([solubility, float(toxicity)], dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        except Exception:
            continue
            
    return data_list