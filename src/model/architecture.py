"""TriLevelFinancialModel: shared encoder + three task heads.

Backbone-agnostic multi-head architecture. The encoder is loaded via
AutoModel so any HuggingFace backbone (RoBERTa, FinBERT, etc.) drops in
without code changes. Macro and Industry heads consume the [CLS] pooled
representation (sentence-level classification). Entity head consumes the
full token sequence (token-level NER).
"""

import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TriLevelFinancialModel(nn.Module):
    def __init__(self, model_name, num_macro, num_industry, num_entity_tags):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        drop = getattr(self.config, "hidden_dropout_prob", 0.1)

        self.macro_head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, num_macro))
        self.industry_head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, num_industry))
        self.entity_head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, num_entity_tags))

    def forward(self, input_ids, attention_mask):
        # token_type_ids omitted so this is universal across BERT/RoBERTa.
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]         # [CLS] for sentence-level
        seq = out.last_hidden_state                   # full seq for NER
        return {
            "macro_logits": self.macro_head(cls),
            "industry_logits": self.industry_head(cls),
            "entity_logits": self.entity_head(seq),
        }
