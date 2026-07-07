import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.data import Batch
from transformers import AutoModel, AutoTokenizer

from util.dataset_smiles import SmilesDataset
from util.gcn_molclr import GCN


class SmilesCNN(nn.Module):
    def __init__(self, input_dim=768, num_filters=64, kernel_sizes=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, k) for k in kernel_sizes
        ])
        self.output_dim = num_filters * len(kernel_sizes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_list = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        return self.dropout(torch.cat(x_list, dim=1))


class DrugPairAttentionAggregator(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.interaction_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, drug1_feat, drug2_feat, cell_feat):
        diff_feat = torch.abs(drug1_feat - drug2_feat)
        prod_feat = drug1_feat * drug2_feat
        interaction_feat = self.interaction_mlp(
            torch.cat([drug1_feat, drug2_feat, diff_feat, prod_feat], dim=-1)
        )

        kv = torch.stack([drug1_feat, drug2_feat], dim=1)
        q = cell_feat.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)

        pair_feat = self.out_proj(torch.cat([interaction_feat, attn_out], dim=-1))
        pair_feat = self.out_norm(pair_feat + interaction_feat)
        return pair_feat


class DualStreamLateAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, tokens):
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused = self.proj(attn_out + tokens)
        return fused.mean(dim=1)


class MKG(nn.Module):
    def __init__(self, args, n_entitys, n_relations, e_dim, r_dim,
                 adj_entity, adj_relation, smiles_seq=None,
                 agg_method='Bi-Interaction',
                 drug_emb_path=None, cell_emb_path=None,
                 chemberta_dir=None, smiles_path=None,
                 cell_gene_path=None):
        super().__init__()

        self.e_dim = e_dim
        self.r_dim = r_dim
        self.agg_method = agg_method
        self.use_smiles = getattr(args, "use_smiles", True)
        self.use_gene = cell_gene_path is not None
        if not self.use_smiles:
            raise ValueError("MKG currently requires model.use_smiles=true.")

        self.tune_chemberta = False

        attn_heads = getattr(args, "attn_heads", 4)
        fusion_dropout = getattr(args, "fusion_dropout", 0.1)
        pred_dropout = getattr(args, "pred_dropout", 0.2)

        self.entity_embs = nn.Embedding(n_entitys, e_dim)
        self.relation_embs = nn.Embedding(n_relations, r_dim, max_norm=1)

        self.adj_entity_cpu = torch.LongTensor(adj_entity)
        self.adj_relation_cpu = torch.LongTensor(adj_relation)
        print("Initializing graph structure...")
        self._init_graph_structure()

        if self.use_smiles:
            if chemberta_dir is None or smiles_path is None:
                raise ValueError("Must provide both chemberta_dir and smiles_path when use_smiles=True.")

            self.tokenizer = AutoTokenizer.from_pretrained(chemberta_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(chemberta_dir, local_files_only=True)

            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

            self.smiles_dim = self.model.config.hidden_size
            self.smiles_cnn = SmilesCNN(input_dim=self.smiles_dim)
            self.cnn_dim = self.smiles_cnn.output_dim
            self.smiles_norm = nn.LayerNorm(self.cnn_dim)

            smiles_df = pd.read_csv(smiles_path)
            if "smiles" not in smiles_df.columns or "entity_id" not in smiles_df.columns:
                raise ValueError("smiles_path must contain columns: ['entity_id', 'smiles'].")

            self.smiles_seq = smiles_df["smiles"].tolist()
            self.entity2smiles = {
                int(row["entity_id"]): idx for idx, row in smiles_df.iterrows()
            }

            self.gnn_graphs = SmilesDataset(smiles_path)
            self.gnn_model = GCN(
                num_layer=3,
                emb_dim=300,
                feat_dim=256,
                drop_ratio=0.1,
                pool='mean'
            )
            self.gnn_out_dim = 256

            self.adapter_smiles = nn.Sequential(
                nn.Linear(self.cnn_dim, e_dim),
                nn.LayerNorm(e_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout)
            )
            self.adapter_gnn = nn.Sequential(
                nn.Linear(self.gnn_out_dim, e_dim),
                nn.LayerNorm(e_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout)
            )
            self.drug_view_attn = nn.MultiheadAttention(
                embed_dim=e_dim,
                num_heads=attn_heads,
                dropout=fusion_dropout,
                batch_first=True
            )
            self.drug_view_ffn = nn.Sequential(
                nn.Linear(e_dim, e_dim),
                nn.LayerNorm(e_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout)
            )
            self.drug_view_norm = nn.LayerNorm(e_dim)

        if drug_emb_path is None or cell_emb_path is None:
            raise ValueError("Must provide both drug_emb_path and cell_emb_path.")

        print(">>> Integrating offline pre-trained features:")
        print(f"  - Drug: {drug_emb_path}")
        print(f"  - Cell: {cell_emb_path}")

        feat_dim = 768

        drug_data = torch.load(drug_emb_path, map_location="cpu") if isinstance(drug_emb_path, str) else drug_emb_path
        drug_matrix = torch.zeros((n_entitys, feat_dim))
        if isinstance(drug_data, dict):
            for eid, emb in drug_data.items():
                eid = int(eid)
                if eid < n_entitys:
                    drug_matrix[eid] = emb.detach().cpu()
        else:
            drug_matrix = drug_data
        self.register_buffer("drug_bank", drug_matrix.float())

        cell_data = torch.load(cell_emb_path, map_location="cpu") if isinstance(cell_emb_path, str) else cell_emb_path
        cell_matrix = torch.zeros((n_entitys, feat_dim))
        if isinstance(cell_data, dict):
            for eid, emb in cell_data.items():
                eid = int(eid)
                if eid < n_entitys:
                    cell_matrix[eid] = emb.detach().cpu()
        else:
            cell_matrix = cell_data
        self.register_buffer("cell_bank", cell_matrix.float())

        self.drug_projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, e_dim),
            nn.LayerNorm(e_dim)
        )
        self.cell_projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, e_dim),
            nn.LayerNorm(e_dim)
        )

        self.txt_norm = nn.LayerNorm(e_dim)
        self.cell_norm = nn.LayerNorm(e_dim)

        if self.use_gene:
            print(f">>> Integrating gene expression: {cell_gene_path}")
            if cell_gene_path.endswith(".csv"):
                gene_df = pd.read_csv(cell_gene_path, header=None)
                gene_dim = gene_df.shape[1] - 1
                gene_matrix = torch.zeros((n_entitys, gene_dim))
                for _, row in gene_df.iterrows():
                    eid = int(row[0])
                    if eid < n_entitys:
                        gene_matrix[eid] = torch.tensor(row[1:].values, dtype=torch.float32)
            else:
                gene_data = torch.load(cell_gene_path, map_location="cpu")
                if isinstance(gene_data, dict):
                    gene_dim = len(list(gene_data.values())[0])
                    gene_matrix = torch.zeros((n_entitys, gene_dim))
                    for eid, emb in gene_data.items():
                        eid = int(eid)
                        if eid < n_entitys:
                            gene_matrix[eid] = emb.detach().cpu()
                else:
                    gene_matrix = gene_data
                    gene_dim = gene_matrix.shape[1]

            self.register_buffer("gene_bank", gene_matrix.float())

            self.gene_projector = nn.Sequential(
                nn.Linear(gene_dim, 512),
                nn.LayerNorm(512),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(512, e_dim),
                nn.LayerNorm(e_dim)
            )
            self.gene_norm = nn.LayerNorm(e_dim)

        self.feature_pair_aggregator = DrugPairAttentionAggregator(
            dim=e_dim, num_heads=attn_heads, dropout=fusion_dropout
        )
        self.knowledge_pair_aggregator = DrugPairAttentionAggregator(
            dim=e_dim, num_heads=attn_heads, dropout=fusion_dropout
        )
        self.late_attention = DualStreamLateAttention(
            dim=e_dim, num_heads=attn_heads, dropout=fusion_dropout
        )

        self.final_predictor = nn.Sequential(
            nn.Linear(e_dim, 512),
            nn.ReLU(),
            nn.Dropout(pred_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(pred_dropout),
            nn.Linear(256, 1)
        )
        self.aux_predictor_feature = nn.Sequential(
            nn.Linear(e_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.aux_predictor_knowledge = nn.Sequential(
            nn.Linear(e_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_embs.weight)
        nn.init.xavier_uniform_(self.relation_embs.weight)

    def _init_graph_structure(self):
        n_nodes, n_neighbors = self.adj_entity_cpu.shape
        src_nodes = torch.arange(n_nodes).unsqueeze(1).expand(-1, n_neighbors).reshape(-1)
        dst_nodes = self.adj_entity_cpu.reshape(-1)
        relations = self.adj_relation_cpu.reshape(-1)
        self.register_buffer("kg_edge_index", torch.stack([src_nodes, dst_nodes], dim=0))
        self.register_buffer("kg_edge_type", relations)

    def _device(self):
        return self.entity_embs.weight.device

    def smiles_embed(self, drug_ids):
        drug_ids = drug_ids.view(-1).long().cpu()
        smiles_indices = [self.entity2smiles.get(int(d.item()), -1) for d in drug_ids]
        smiles_batch = [
            self.smiles_seq[idx] if idx != -1 else self.smiles_seq[0]
            for idx in smiles_indices
        ]

        inputs = self.tokenizer(
            smiles_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self._device())

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        h = self.smiles_cnn(outputs.last_hidden_state)
        h = self.smiles_norm(h)
        return h

    def gnn_embed(self, drug_ids):
        drug_ids = drug_ids.view(-1).long().cpu()
        smiles_indices = [self.entity2smiles.get(int(d.item()), -1) for d in drug_ids]
        graphs = [self.gnn_graphs[idx] if idx != -1 else self.gnn_graphs[0] for idx in smiles_indices]
        batch = Batch.from_data_list(graphs).to(self._device())
        h, _ = self.gnn_model(batch)
        return h

    def encode_drug_feature(self, drug_ids):
        h_smiles = self.adapter_smiles(self.smiles_embed(drug_ids))
        h_gnn = self.adapter_gnn(self.gnn_embed(drug_ids))

        views = torch.stack([h_smiles, h_gnn], dim=1)
        attn_out, _ = self.drug_view_attn(views, views, views)
        views = self.drug_view_norm(views + attn_out)
        views = self.drug_view_ffn(views) + views
        return views.mean(dim=1)

    def forward(self, u1, u2, c):
        u1 = u1.long()
        u2 = u2.long()
        c = c.long()

        h_mol_1 = self.encode_drug_feature(u1)
        h_mol_2 = self.encode_drug_feature(u2)

        if self.use_gene:
            h_gene = self.gene_norm(self.gene_projector(self.gene_bank[c]))
            pair_feat_feature = self.feature_pair_aggregator(h_mol_1, h_mol_2, h_gene)
        else:
            h_gene = None
            zero_cell = torch.zeros_like(h_mol_1)
            pair_feat_feature = self.feature_pair_aggregator(h_mol_1, h_mol_2, zero_cell)

        h_txt_1 = self.txt_norm(self.drug_projector(self.drug_bank[u1]))
        h_txt_2 = self.txt_norm(self.drug_projector(self.drug_bank[u2]))
        h_cell_context = self.cell_norm(self.cell_projector(self.cell_bank[c]))

        pair_feat_knowledge = self.knowledge_pair_aggregator(
            h_txt_1, h_txt_2, h_cell_context
        )

        if self.use_gene:
            final_tokens = torch.stack(
                [pair_feat_feature, pair_feat_knowledge, h_gene, h_cell_context],
                dim=1
            )
        else:
            final_tokens = torch.stack(
                [pair_feat_feature, pair_feat_knowledge, h_cell_context],
                dim=1
            )

        final_repr = self.late_attention(final_tokens)
        raw_logits = self.final_predictor(final_repr).squeeze(-1)

        if self.training:
            logits_feature = self.aux_predictor_feature(pair_feat_feature).squeeze(-1)
            logits_knowledge = self.aux_predictor_knowledge(pair_feat_knowledge).squeeze(-1)
            return raw_logits, pair_feat_feature, pair_feat_knowledge, logits_feature, logits_knowledge

        return raw_logits
