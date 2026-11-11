# models/word_model.py
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    T5Config, MT5Config, AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
import logging

def get_t5_hidden_size(model_name_hf, fallback_size):
    """Dynamically determine T5 hidden size (d_model)."""
    try:
        logging.info(f"Attempting to load config for {model_name_hf}...")
        # Try AutoConfig first
        config = AutoConfig.from_pretrained(model_name_hf, trust_remote_code=True)
        logging.info(f"Successfully loaded config. d_model: {config.d_model}")
        return config.d_model
    except Exception as e:
        logging.warning(f"Could not load config using AutoConfig for {model_name_hf}. Error: {e}. Trying specific classes...")
        try:
            if "mt5" in model_name_hf.lower():
                config = MT5Config.from_pretrained(model_name_hf)
            else:
                config = T5Config.from_pretrained(model_name_hf)
            logging.info(f"Successfully loaded config. d_model: {config.d_model}")
            return config.d_model
        except Exception as e2:
            logging.error(f"Could not determine hidden size dynamically for {model_name_hf}. Error: {e2}. Using fallback: {fallback_size}")
            return fallback_size

class T5EncoderClassifier(nn.Module):
    def __init__(self, feature_dim, projected_dim, model_name_hf, num_classes, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.model_name_hf = model_name_hf
        self.feature_dim = feature_dim # Dim of input keypoints (e.g., 63 or 126)

        # Load T5 Base Model (Encoder part is used)
        try:
            logging.info(f"Loading T5 base model: {model_name_hf}")
            if "mt5" in model_name_hf.lower():
                self.t5_model = MT5ForConditionalGeneration.from_pretrained(model_name_hf)
            else:
                self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name_hf)
            self.projected_dim = self.t5_model.config.d_model # Use actual loaded model's hidden size
            logging.info(f"T5 model loaded. Using projected_dim (d_model): {self.projected_dim}")

        except Exception as e:
            logging.warning(f"Could not load T5ForConditionalGeneration ({e}), trying AutoModelForSeq2SeqLM...")
            try:
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_hf, trust_remote_code=True)
                self.projected_dim = self.t5_model.config.d_model
                logging.info(f"AutoModel loaded. Using projected_dim (d_model): {self.projected_dim}")
            except Exception as e2:
                 logging.error(f"Failed to load any T5 model ({e2}). Cannot initialize T5EncoderClassifier.")
                 raise RuntimeError(f"Failed to load T5 model {model_name_hf}") from e2

        if self.projected_dim != projected_dim:
             logging.warning(f"Provided projected_dim ({projected_dim}) differs from loaded model's d_model ({self.projected_dim}). Using loaded model's value.")

        # Linear projection layer
        self.custom_linear = nn.Sequential(
            nn.Linear(self.feature_dim, self.projected_dim),
            nn.Dropout(0.1), # Consider matching dropout_rate?
            nn.LayerNorm(self.projected_dim),
            nn.GELU(),
        )
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.projected_dim, num_classes)

    def forward(self, input_features, attention_mask=None, labels=None, **kwargs):
        # input_features shape: [batch, seq_len, feature_dim]
        inputs_embeds = self.custom_linear(input_features) # Project keypoints to T5 hidden size
        # inputs_embeds shape: [batch, seq_len, projected_dim]

        encoder_outputs = self.t5_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # Shape: [batch, seq_len]
            return_dict=True
        )
        sequence_output = encoder_outputs.last_hidden_state # Shape: [batch, seq_len, projected_dim]

        # Use first token's output ([CLS] equivalent for sequence classification)
        pooled_output = sequence_output[:, 0] # Shape: [batch, projected_dim]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # Shape: [batch, num_classes]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )