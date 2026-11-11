# models/alphabet_model.py
import torch
import torch.nn as nn
from torchvision import models

class ArSLAttentionLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=2, bidirectional=True, dropout_rate=0.5):
        super(ArSLAttentionLSTM, self).__init__()

        # Load ResNet18 with current recommended weights argument
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.feature_extractor = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2]) # Remove fc and avgpool

        self.reshape_size = 512 # Output channels of ResNet18 block before avgpool
        self.seq_length = 49  # Spatial dimensions flattened (7*7 for 224x224 input)

        self.lstm = nn.LSTM(
            input_size=self.reshape_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), # BatchNorm might need handling for batch_size=1 in eval mode
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256), # BatchNorm might need handling for batch_size=1 in eval mode
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)  # [batch, 512, 7, 7]
        x = x.view(batch_size, self.reshape_size, -1).permute(0, 2, 1)  # [batch, 49, 512]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]

        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1).unsqueeze(-1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)

        x = self.dropout(context_vector)

        # Classifier expects [batch, features]
        # Handle BatchNorm issue with batch_size=1 during eval
        # model.eval() should handle this, but explicit check can be added if issues persist
        if batch_size == 1 and (isinstance(self.classifier[2], nn.BatchNorm1d) or isinstance(self.classifier[6], nn.BatchNorm1d)):
             # In eval mode, BatchNorm uses running stats, should be okay.
             # If training mode, or issues arise, might need to skip BN or use InstanceNorm.
             pass # Assuming eval mode handles it

        x = self.classifier(x)
        return x