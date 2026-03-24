import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TriLevelFinancialModel(nn.Module):
    def __init__(self, model_name, num_macro, num_industry, num_entity_tags):
        super(TriLevelFinancialModel, self).__init__()
        
        # 1. Dynamically load the configuration and the model
        # AutoConfig automatically finds the hidden dimension size (usually 768)
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size
        
        # 2. Level 1: Macro Head (Multi-label)
        self.macro_head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob if hasattr(self.config, 'hidden_dropout_prob') else 0.1),
            nn.Linear(hidden_size, num_macro)
        )
        
        # 3. Level 2: Industry Head (Multi-label or Multi-class)
        self.industry_head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob if hasattr(self.config, 'hidden_dropout_prob') else 0.1),
            nn.Linear(hidden_size, num_industry)
        )
        
        # 4. Level 3: Entity Head (NER Token Classification)
        self.entity_head = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob if hasattr(self.config, 'hidden_dropout_prob') else 0.1),
            nn.Linear(hidden_size, num_entity_tags)
        )

    def forward(self, input_ids, attention_mask):
        # Note: We do NOT pass token_type_ids. 
        # RoBERTa don't use them, so omitting them makes the code universal.
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Universal way to get the [CLS] token representation across all 4 models
        # We take the first token (index 0) from the last hidden state
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        
        # Sequence output for NER (every single token)
        sequence_output = outputs.last_hidden_state
        
        # Pass through heads
        macro_logits = self.macro_head(cls_token_output)
        industry_logits = self.industry_head(cls_token_output)
        entity_logits = self.entity_head(sequence_output)
        
        return {
            "macro_logits": macro_logits,
            "industry_logits": industry_logits,
            "entity_logits": entity_logits
        }