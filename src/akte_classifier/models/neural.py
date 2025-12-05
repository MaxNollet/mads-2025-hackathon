import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class TextVectorizer:
    """
    Wraps a HuggingFace model to vectorize text.
    """

    def __init__(
        self, model_name: str, max_length: int | None = None, pooling: str | None = None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to eval mode by default

        if pooling is None:
            self.pooling = self._detect_pooling_strategy(model_name)
        else:
            self.pooling = pooling

        # Determine max_length
        if max_length is not None:
            self.max_length = max_length
        else:
            # Try to get from tokenizer
            model_max_len = self.tokenizer.model_max_length
            # HuggingFace tokenizers often return a very large int if not set
            # We treat anything > 10000 as "infinite"/unset and fallback to 512
            if model_max_len > 10000:
                logger.warning(
                    f"Tokenizer model_max_length is {model_max_len}, falling back to 512. "
                    "Specify max_length explicitly if needed."
                )
                self.max_length = 512
            else:
                self.max_length = model_max_len

        logger.info(
            f"TextVectorizer initialized with max_length={self.max_length}, pooling={self.pooling}"
        )

    def _detect_pooling_strategy(self, model_name: str) -> str:
        """
        Attempts to detect the pooling strategy from the model's configuration.
        """
        try:
            # Download modules.json
            path = hf_hub_download(model_name, "modules.json")
            with open(path) as f:
                modules = json.load(f)

            # Look for the pooling module
            for module in modules:
                if module["type"] == "sentence_transformers.models.Pooling":
                    # Download the pooling config
                    config_path = hf_hub_download(
                        model_name, f"{module['path']}/config.json"
                    )
                    with open(config_path) as f:
                        config = json.load(f)

                    if config.get("pooling_mode_cls_token"):
                        logger.success(
                            f"Auto-detected pooling strategy: cls (from {model_name})"
                        )
                        return "cls"
                    if config.get("pooling_mode_mean_tokens"):
                        logger.success(
                            f"Auto-detected pooling strategy: mean (from {model_name})"
                        )
                        return "mean"
        except Exception:
            # Fail silently on network errors or missing files
            pass

        logger.warning(
            f"Could not detect pooling strategy for {model_name}, defaulting to 'mean'. "
            "Please check the model card to see if 'cls' pooling is required."
        )
        return "mean"

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of texts into vectors.
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == "cls":
            # CLS token is usually the first token
            return outputs.last_hidden_state[:, 0, :]

        # Default to mean pooling
        # batches of text are padded such that they have the same length,
        # however, the padding tokens are zero and we dont want to include them in the mean
        attention_mask = inputs["attention_mask"]
        # attention_mask shape: (batch_size, seq_len)
        token_embeddings = outputs.last_hidden_state
        # last_hidden_state shape: (batch_size, seq_len, hidden_dim)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Zero out padding tokens so they don't skew the sum, then
        # divide by the count of real tokens (not total length) for a true average.
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
'''

class TextVectorizer(nn.Module): # Let op: inherit nu van nn.Module voor de extra weights
    """
    Wraps a HuggingFace model with Attention Pooling.
    """

    def __init__(
        self, model_name: str, max_length: int | None = None, pooling: str | None = None
    ):
        super().__init__() # Belangrijk voor nn.Module
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # --- Nieuw: Attention Weights ---
        # We leren een vector die bepaalt hoe belangrijk elk hidden state is
        self.attention_weights = nn.Linear(self.model.config.hidden_size, 1)
        
        # Determine max_length logic (hetzelfde als je had)
        if max_length is not None:
            self.max_length = max_length
        else:
            model_max_len = self.tokenizer.model_max_length
            self.max_length = 512 if model_max_len > 10000 else model_max_len

    def forward(self, texts: list[str]) -> torch.Tensor:
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass door BERT
        outputs = self.model(**inputs)
        
        # last_hidden_state: (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state 
        attention_mask = inputs["attention_mask"] # (batch, seq_len)

        # --- Attention Pooling Logic ---
        # 1. Bereken raw attention scores voor elk woord
        # shape: (batch, seq_len, 1)
        attn_scores = self.attention_weights(hidden_states) 
        
        # 2. Maskeer padding tokens (geef ze -oneindig score zodat softmax ze negeert)
        attn_scores = attn_scores.squeeze(-1) # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # 3. Softmax om kansverdeling te krijgen (alles telt op tot 1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1) # (batch, seq_len, 1)
        
        # 4. Gewogen som van de hidden states
        # (batch, seq_len, hidden) * (batch, seq_len, 1) -> sum over seq_len
        weighted_embeddings = torch.sum(hidden_states * attn_weights, dim=1)
        
        return weighted_embeddings

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
'''

class NeuralClassifier(nn.Module):
    """
    A simple neural network for classification.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns logits.
        """
        x = self.sequential(x)
        return x

import torch
import torch.nn as nn
from loguru import logger

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3): # Dropout verhoogd!
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HybridClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        regex_dim: int, 
        num_classes: int, 
        hidden_dim: int = 128,  # Terug naar 128 (compacter)
        num_res_blocks: int = 1  # Terug naar 1 (minder complexiteit)
    ):
        super().__init__()
        
        # Branch dimensions
        text_hidden = hidden_dim
        regex_hidden = hidden_dim

        # --- 1. Tekst Branch ---
        self.text_input = nn.Sequential(
            nn.Linear(input_dim, text_hidden),
            nn.BatchNorm1d(text_hidden),
            nn.ReLU(),
            nn.Dropout(0.3) # Extra dropout aan het begin
        )
        self.text_res_blocks = nn.Sequential(
            *[ResidualBlock(text_hidden, dropout=0.3) for _ in range(num_res_blocks)]
        )

        # --- 2. Regex Branch ---
        self.regex_input = nn.Sequential(
            nn.Linear(regex_dim, regex_hidden),
            nn.BatchNorm1d(regex_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regex_res_blocks = nn.Sequential(
            *[ResidualBlock(regex_hidden, dropout=0.3) for _ in range(num_res_blocks)]
        )

        # --- 3. Fusion ---
        combined_dim = text_hidden + regex_hidden
        
        # Fusion Layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim), # Belangrijk: Normaliseer na samenvoegen
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3), # Hoge dropout voor de finale beslissing
            nn.Linear(combined_dim // 2, num_classes)
        )

    def forward(
        self, text_emb: torch.Tensor, regex_feats: torch.Tensor
    ) -> torch.Tensor:
        
        x_text = self.text_input(text_emb)
        x_text = self.text_res_blocks(x_text)

        x_regex = self.regex_input(regex_feats)
        x_regex = self.regex_res_blocks(x_regex)

        # Concatenate
        combined = torch.cat((x_text, x_regex), dim=1)

        logits = self.classifier(combined)
        return logits
'''
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HybridClassifier(nn.Module):
    """
    Gated Fusion Hybrid Classifier.
    """
    def __init__(
        self, 
        input_dim: int, 
        regex_dim: int, 
        num_classes: int, 
        hidden_dim: int = 128, 
        num_res_blocks: int = 1
    ):
        super().__init__()
        
        # Branch dimensions
        text_hidden = hidden_dim
        regex_hidden = hidden_dim

        # --- 1. Tekst Branch (Deze ontbrak net) ---
        self.text_input = nn.Sequential(
            nn.Linear(input_dim, text_hidden),
            nn.BatchNorm1d(text_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_res_blocks = nn.Sequential(
            *[ResidualBlock(text_hidden, dropout=0.3) for _ in range(num_res_blocks)]
        )

        # --- 2. Regex Branch (Deze ontbrak net) ---
        self.regex_input = nn.Sequential(
            nn.Linear(regex_dim, regex_hidden),
            nn.BatchNorm1d(regex_hidden),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.regex_res_blocks = nn.Sequential(
            *[ResidualBlock(regex_hidden, dropout=0.3) for _ in range(num_res_blocks)]
        )

        # --- 3. Gated Fusion ---
        combined_dim = text_hidden + regex_hidden
        
        # De 'Gate' berekent een weging tussen 0 en 1 voor elk feature
        self.gate_layer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_dim // 2, num_classes)
        )

    def forward(self, text_emb: torch.Tensor, regex_feats: torch.Tensor) -> torch.Tensor:
        # 1. Branches verwerken
        x_text = self.text_input(text_emb)
        x_text = self.text_res_blocks(x_text)

        x_regex = self.regex_input(regex_feats)
        x_regex = self.regex_res_blocks(x_regex)

        # 2. Concatenate
        combined_raw = torch.cat((x_text, x_regex), dim=1)
        
        # 3. Gated Mechanism
        # We kijken naar de gecombineerde features en bepalen wat we belangrijk vinden
        gate = self.gate_layer(combined_raw)
        
        # Element-wise vermenigvuldiging: filtert ruis weg op basis van de gate
        gated_features = combined_raw * gate 
        
        # 4. Classify
        logits = self.classifier(gated_features)
        return logits
'''