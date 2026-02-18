#extracted from https://huggingface.co/abhilash88/age-gender-prediction/blob/main/model.py

import torch
import torch.nn as nn
from transformers import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class AgeGenderViTModel(ViTPreTrainedModel):
    """
    Age-Gender Vision Transformer Model
    Architecture: ViT-Base with dual heads for age and gender prediction
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        
        # Age regression head: 768 -> 256 -> 64 -> 1
        self.age_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Gender classification head: 768 -> 256 -> 64 -> 1 (sigmoid)
        self.gender_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
        # Classifier for pipeline compatibility
        self.num_labels = 2
        self.config.num_labels = 2
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.post_init()
        
    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = self.vit(pixel_values=pixel_values, **kwargs)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # CLS token
        
        age_output = self.age_head(pooled_output)
        gender_output = self.gender_head(pooled_output)
        
        # Concatenate age and gender for custom processing
        logits = torch.cat([age_output, gender_output], dim=1)
        
        # Store in output for postprocessing
        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )