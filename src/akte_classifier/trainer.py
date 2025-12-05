import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from akte_classifier.datasets.dataset import DatasetFactory
from akte_classifier.models.llm import LLMClassifier
from akte_classifier.models.neural import (HybridClassifier, NeuralClassifier,
                                           TextVectorizer)
# VOEG DEZE REGEL TOE:
from akte_classifier.models.nebius import NebiusTextVectorizer
from akte_classifier.models.prompts import ClassificationPromptTemplate
from akte_classifier.models.regex import RegexGenerator, RegexVectorizer
from akte_classifier.utils.data import get_long_tail_labels, load_descriptions
from akte_classifier.utils.early_stopping import EarlyStopping
from akte_classifier.utils.evaluation import Evaluator
from akte_classifier.utils.logging import (CompositeLogger, ConsoleLogger,
                                           MLFlowLogger)


def get_default_device() -> str:
    device = os.environ.get("DEVICE")
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainingConfig:
    data_path: str = "assets/train.jsonl"
    csv_path: str = "assets/rechtsfeiten.csv"
    batch_size: int = 32
    split_ratio: float = 0.8
    model_name: str = "prajjwal1/bert-tiny"
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = get_default_device()
    model_class: str = "HybridClassifier"  # Default to Hybrid
    use_regex: bool = True  # Whether to use regex features
    hidden_dim: int = 512  # Hidden layer dimension
    max_length: Optional[int] = None  # Max token length (None = auto)
    pooling: Optional[str] = None  # Pooling strategy: "mean", "cls", or None (auto)
    long_tail_threshold: Optional[int] = None  # Threshold for long-tail labels
    experiment_name: str = "kadaster_experiment"  # MLFlow experiment name
    patience: int = 10
    min_delta: float = 0.001


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        logger.success(f"Using device: {self.device}")
        self.loaders: Dict[str, DataLoader] = {}
        self.num_classes = 0
        self.class_names: List[str] = []

        # Models
        self.vectorizer: Optional[TextVectorizer] = None
        self.regex_vectorizer: Optional[RegexVectorizer] = None
        self.classifier: Optional[nn.Module] = None

        # training
        self.loss_fn: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # evaluation
        self.evaluator: Optional[Evaluator] = None
        self.logger = CompositeLogger([ConsoleLogger(), MLFlowLogger()])

        # Checkpointing
        self.best_model_path: Optional[str] = None
        self.best_codes_path: Optional[str] = None
        self.best_config_path: Optional[str] = None
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, delta=self.config.min_delta
        )

    def get_data(self):
        logger.info("Initializing DatasetFactory...")
        factory = DatasetFactory(
            self.config.data_path,
            batch_size=self.config.batch_size,
            split_ratio=self.config.split_ratio,
        )

        # Initialize Text Vectorizer
        if self.config.model_class in ["NeuralClassifier", "HybridClassifier"]:
            # CHECK: Gebruiken we een Nebius/API model?
            if "nebius" in self.config.model_name.lower() or "bge" in self.config.model_name.lower():
                logger.info(f"Initializing Nebius API Vectorizer for {self.config.model_name}...")
                self.vectorizer = NebiusTextVectorizer(model_name=self.config.model_name)
                self.vectorizer.to(self.device)
            else:
                # OUDE LOGICA: Lokaal HuggingFace model
                self.vectorizer = TextVectorizer(
                    self.config.model_name,
                    max_length=self.config.max_length,
                    pooling=self.config.pooling,
                )
                self.vectorizer.model.to(self.device)
                for param in self.vectorizer.model.parameters():
                    param.requires_grad = False
            
            logger.info("Initialized TextVectorizer...")
        else:
            logger.info("Skipping TextVectorizer initialization...")
            self.vectorizer = None

        # Initialize Regex Vectorizer if using Hybrid or RegexOnly
        if self.config.use_regex:
            logger.info("Initializing RegexVectorizer...")
            regex_gen = RegexGenerator(self.config.csv_path)
            # Pass the encoder to ensure alignment
            self.regex_vectorizer = RegexVectorizer(
                regex_gen, label_encoder=factory.encoder
            )

        self.loaders = factory.get_vectorized_loader(
            self.vectorizer, self.regex_vectorizer
        )

        self.num_classes = len(factory.encoder)
        self.encoder_codes = factory.encoder.unique_codes
        self.class_names = [
            str(factory.encoder.idx2code.get(i + 1, i + 1))
            for i in range(self.num_classes)
        ]
        logger.info(f"Number of classes: {self.num_classes}")

        # Initialize Evaluator
        self.evaluator = Evaluator(self.num_classes, class_names=self.class_names)

    def setup_models(self):
        logger.info(f"Initializing {self.config.model_class}...")

        if self.config.model_class == "NeuralClassifier":
            self.classifier = NeuralClassifier(
                input_dim=self.vectorizer.hidden_size,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        elif self.config.model_class == "HybridClassifier":
            self.classifier = HybridClassifier(
                input_dim=self.vectorizer.hidden_size,
                regex_dim=self.regex_vectorizer.output_dim,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        elif self.config.model_class == "RegexOnlyClassifier":
            # Ensure regex vectorizer is available
            if not self.regex_vectorizer:
                raise ValueError(
                    "RegexOnlyClassifier requires regex features. Ensure regex_vectorizer is initialized."
                )

            self.classifier = NeuralClassifier(
                input_dim=self.regex_vectorizer.output_dim,
                num_classes=self.num_classes,
                hidden_dim=self.config.hidden_dim,
            )
        else:
            raise ValueError(f"Unknown model class: {self.config.model_class}")

        self.classifier.to(self.device)

    def setup_optimization(self):
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.classifier.parameters(), lr=self.config.learning_rate
        )

    def train_epoch(self, epoch: int):
        assert self.classifier is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None

        self.classifier.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.loaders["train"], desc=f"Epoch {epoch}/{self.config.num_epochs}"
        )

        for batch in progress_bar:
            # Handle variable number of items from loader
            if len(batch) == 3:
                embeddings, labels, regex_feats = batch
                embeddings = embeddings.to(self.device)
                regex_feats = regex_feats.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if self.config.model_class == "HybridClassifier":
                    logits = self.classifier(embeddings, regex_feats)
                elif self.config.model_class == "RegexOnlyClassifier":
                    logits = self.classifier(regex_feats)
                else:
                    # NeuralClassifier ignores regex features if present (shouldn't happen with correct loader but safe fallback)
                    logits = self.classifier(embeddings)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.classifier(embeddings)

            # Compute loss
            loss = self.loss_fn(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.loaders["train"])
        self.logger.log_metrics({"train_loss": avg_loss}, step=epoch)

    def validate(self, epoch: int) -> Dict[str, Any]:
        assert self.classifier is not None
        assert self.loss_fn is not None
        assert self.evaluator is not None

        self.classifier.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.loaders["test"]:
                if len(batch) == 3:
                    embeddings, labels, regex_feats = batch
                    embeddings = embeddings.to(self.device)
                    regex_feats = regex_feats.to(self.device)
                    labels = labels.to(self.device)

                    if self.config.model_class == "HybridClassifier":
                        logits = self.classifier(embeddings, regex_feats)
                    elif self.config.model_class == "RegexOnlyClassifier":
                        logits = self.classifier(regex_feats)
                    else:
                        logits = self.classifier(embeddings)
                else:
                    embeddings, labels = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.classifier(embeddings)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # Probabilities
                probs = torch.sigmoid(logits)
                # Predictions (threshold 0.5)
                preds = (probs > 0.5).float()

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.loaders["test"])

        # Concatenate
        all_preds_arr = np.vstack(all_preds)
        all_probs_arr = np.vstack(all_probs)
        all_targets_arr = np.vstack(all_targets)

        # Compute Metrics using Evaluator
        metrics = self.evaluator.compute_metrics(all_targets_arr, all_preds_arr)
        metrics["val_loss"] = avg_loss

        self.logger.log_metrics(metrics, step=epoch)
        return {
            "preds": all_preds_arr,
            "probs": all_probs_arr,
            "targets": all_targets_arr,
            "val_loss": avg_loss,
            **metrics,
        }

    def run(self):
        logger.info("Start setup...")
        self.get_data()
        self.setup_models()
        self.setup_optimization()

        self.logger.log_params(asdict(self.config))
        logger.success("initialized run")

        # Log model tags (hashes)
        tags = {}

        # Only log text model name if we are using it (Neural or Hybrid)
        if self.config.model_class != "RegexOnlyClassifier":
            tags["text_model_name"] = self.vectorizer.model_name

        if self.regex_vectorizer:
            tags["regex_hash"] = self.regex_vectorizer.hash

        self.logger.log_tags(tags)

        logger.info("Start training...")

        for epoch in range(1, self.config.num_epochs + 1):
            self.train_epoch(epoch)
            val_results = self.validate(epoch)

            # Early Stopping check
            self.early_stopping(val_results["val_loss"])

            if self.early_stopping.improved:
                # Save new best model (overwrites existing file due to fixed timestamp)
                self.best_model_path, self.best_codes_path, self.best_config_path = (
                    self.save_checkpoint(epoch, tags)
                )

            if self.early_stopping.early_stop:
                logger.warning("Early stopping triggered!")
                break

            # Plot artifacts at the end
            if epoch == self.config.num_epochs:
                self.evaluator.plot_overview_metrics(
                    val_results["targets"], val_results["preds"], tags=tags
                )
                self.evaluator.plot_roc_curve(
                    val_results["targets"], val_results["probs"], tags=tags
                )
                self.evaluator.plot_pr_curve(
                    val_results["targets"], val_results["probs"], tags=tags
                )
                self.evaluator.save_per_class_metrics(
                    val_results["targets"], val_results["preds"], tags=tags
                )

    def save_checkpoint(self, epoch: int, tags: Dict[str, str]):
        """
        Saves model weights and encoder codes.
        """
        # Create artifacts/models directory
        model_dir = "artifacts/models"
        os.makedirs(model_dir, exist_ok=True)

        timestamp = self.run_timestamp

        # Construct model slug
        if self.config.model_class == "RegexOnlyClassifier":
            model_slug = "regex_only"
        else:
            model_slug = self.config.model_name.replace("/", "_")

        # Add regex hash if available
        if "regex_hash" in tags:
            model_slug += f"_{tags['regex_hash']}"

        filename_base = f"{model_slug}_{timestamp}"

        # 1. Save Model Weights
        model_path = f"{model_dir}/{filename_base}.pt"
        assert self.classifier is not None
        torch.save(self.classifier.state_dict(), model_path)
        logger.info(f"Saved model checkpoint to {model_path}")

        # 2. Save Encoder Codes
        if self.loaders and "train" in self.loaders:
            pass

        codes_path = None
        if hasattr(self, "encoder_codes"):
            codes_path = f"{model_dir}/{filename_base}_codes.json"
            with open(codes_path, "w") as f:
                json.dump(self.encoder_codes, f)
            logger.info(f"Saved encoder codes to {codes_path}")
        else:
            logger.warning("Encoder codes not available, skipping codes saving.")

        # 3. Save Training Config
        config_path = f"{model_dir}/{filename_base}_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=4)
        logger.success(f"Saved training config to {config_path}")

        return model_path, codes_path, config_path

    def evaluate_file(
        self,
        file_path: str,
        model_path: str,
        codes_path: str,
        csv_path: str = "assets/rechtsfeiten.csv",
    ):
        """
        Evaluates a trained model on a specific file.
        """
        logger.info(f"Loading encoder codes from {codes_path}...")
        with open(codes_path, "r") as f:
            encoder_codes = json.load(f)
        logger.info(f"Loaded {len(encoder_codes)} encoder codes.")

        self.regex_vectorizer = None  # Initialize to None
        if self.config.use_regex:
            logger.info("Initializing Regex Model for evaluation...")
            from akte_classifier.datasets.dataset import LabelEncoder
            from akte_classifier.models.regex import (RegexGenerator,
                                                      RegexVectorizer)

            generator = RegexGenerator(csv_path)
            # RegexVectorizer needs the encoder to map regex matches to indices.
            encoder = LabelEncoder(encoder_codes)

            self.regex_vectorizer = RegexVectorizer(generator, label_encoder=encoder)

        factory = DatasetFactory(
            file_path=file_path,
            batch_size=self.config.batch_size,
            split_ratio=1.0,  # Use full file for evaluation
            encoder_codes=encoder_codes,
        )

        # 3. Initialize Models
        if self.config.model_class in ["NeuralClassifier", "HybridClassifier"]:
            # CHECK: API vs Lokaal (dezelfde logica als hierboven)
            if "nebius" in self.config.model_name.lower() or "bge" in self.config.model_name.lower():
                logger.info(f"Initializing Nebius API Vectorizer for evaluation: {self.config.model_name}")
                self.vectorizer = NebiusTextVectorizer(model_name=self.config.model_name)
                self.vectorizer.to(self.device)
            else:
                self.vectorizer = TextVectorizer(
                    self.config.model_name,
                    max_length=self.config.max_length,
                    pooling=self.config.pooling,
                )
                self.vectorizer.model.to(self.device)
                for param in self.vectorizer.model.parameters():
                    param.requires_grad = False
        else:
            self.vectorizer = None

        # Regex Vectorizer
        if self.config.use_regex:
            regex_gen = RegexGenerator(self.config.csv_path)
            self.regex_vectorizer = RegexVectorizer(
                regex_gen, label_encoder=factory.encoder
            )

        # 4. Vectorize Data
        assert (
            self.vectorizer is not None
            or self.config.model_class == "RegexOnlyClassifier"
        )
        loaders = factory.get_vectorized_loader(
            self.vectorizer, self.regex_vectorizer, splits=["train"]
        )
        eval_loader = loaders["train"]

        assert factory.encoder is not None, "Encoder must be initialized"
        self.num_classes = len(factory.encoder)
        self.class_names = [
            str(factory.encoder.idx2code.get(i + 1, i + 1))
            for i in range(self.num_classes)
        ]

        # 5. Setup Model and Load Weights
        self.setup_models()

        logger.info(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        assert self.classifier is not None
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

        # 6. Run Evaluation
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(self.num_classes, class_names=self.class_names)
        logger.success("Evaluation setup complete, starting evaluation...")

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                if len(batch) == 3:
                    texts, regex_feats, labels = batch
                    regex_feats = regex_feats.to(self.device)
                else:
                    texts, labels = batch
                    regex_feats = None

                labels = labels.to(self.device)

                emb = texts.to(self.device)

                if self.config.model_class == "RegexOnlyClassifier":
                    assert self.classifier is not None
                    logits = self.classifier(regex_feats)
                elif self.config.model_class == "HybridClassifier":
                    assert self.classifier is not None
                    logits = self.classifier(emb, regex_feats)
                else:
                    assert self.classifier is not None
                    logits = self.classifier(emb)

                assert self.loss_fn is not None
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(eval_loader)

        all_preds_arr = np.vstack(all_preds)
        all_probs_arr = np.vstack(all_probs)
        all_targets_arr = np.vstack(all_targets)

        assert self.evaluator is not None
        metrics = self.evaluator.compute_metrics(all_targets_arr, all_preds_arr)
        metrics["eval_loss"] = avg_loss

        logger.info(f"Evaluation Results: {metrics}")

        import re

        timestamp_match = re.search(r"(\d{8}_\d{6})", os.path.basename(model_path))
        timestamp_prefix = f"{timestamp_match.group(1)}_" if timestamp_match else ""

        # Generate plots/artifacts
        eval_file_name = f"{timestamp_prefix}eval_{os.path.basename(file_path)}"
        tags = {"eval_file": eval_file_name}

        assert self.evaluator is not None
        self.evaluator.save_per_class_metrics(all_targets_arr, all_preds_arr, tags=tags)
        self.evaluator.plot_overview_metrics(all_targets_arr, all_preds_arr, tags=tags)
        self.evaluator.plot_roc_curve(all_targets_arr, all_probs_arr, tags=tags)
        self.evaluator.plot_pr_curve(all_targets_arr, all_probs_arr, tags=tags)
        logger.success(f"Saved evaluation artifacts with {timestamp_prefix}")

        return metrics


class LLMRunner:
    def __init__(
        self,
        threshold: int,
        limit: Optional[int],
        model_name: str,
        experiment_name: str,
        max_length: Optional[int] = None,
    ):
        self.threshold = threshold
        self.limit = limit
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.max_length = max_length

        # Init MLFlow Logger
        self.mlflow_logger = MLFlowLogger(experiment_name=experiment_name)
        self.mlflow_logger.enable_genai_autolog()

    def _load_resources(self):
        """
        Loads long-tail labels, descriptions, dataset, and initializes the classifier.
        """
        # 1. Get long-tail labels
        dist_path = "artifacts/csv/label_distribution.csv"
        long_tail_codes = get_long_tail_labels(dist_path, self.threshold)
        if not long_tail_codes:
            logger.error("No long-tail labels found. Check threshold or file.")
            return None, None, None, None

        logger.info(f"Found {len(long_tail_codes)} long-tail labels.")

        # 2. Get descriptions for these labels
        csv_path = "assets/rechtsfeiten.csv"
        descriptions = load_descriptions(csv_path, long_tail_codes)

        if not descriptions:
            logger.error("No descriptions found for the long-tail labels.")
            return None, None, None, None

        # 3. Init DatasetFactory with filtering
        factory = DatasetFactory(
            file_path="assets/train.jsonl",
            long_tail_threshold=self.threshold,
            batch_size=1,
        )

        if not factory.train_dataset or len(factory.train_dataset) == 0:
            logger.error("No samples found in dataset after filtering.")
            return None, None, None, None

        # 4. Init LLM Classifier
        prompt_template = ClassificationPromptTemplate()
        classifier = LLMClassifier(
            model_name=self.model_name,
            descriptions=descriptions,
            prompt_template=prompt_template,
            max_length=self.max_length,
        )
        logger.success(f"Initialized LLM classifier for {self.model_name}")

        return long_tail_codes, descriptions, factory, classifier

    def _run_inference(self, factory, classifier, long_tail_codes):
        """
        Runs the classification loop.
        """
        logger.info("Classifying samples...")

        # Access the underlying HuggingFace dataset to get raw text
        hf_dataset = factory.train_dataset.dataset

        all_true_labels = []
        all_pred_labels = []
        predictions_data = []

        count = 0

        # Determine total for progress bar
        total_samples = len(hf_dataset)
        if self.limit is not None:
            total_samples = min(self.limit, total_samples)

        for i in tqdm(range(len(hf_dataset)), total=total_samples, desc="Classifying"):
            if self.limit is not None and count >= self.limit:
                break

            sample = hf_dataset[i]
            text = sample["text"]
            akte_id = sample.get("akteId", "unknown")
            true_labels = [int(c) for c in sample["rechtsfeitcodes"]]

            # Only process if it actually has long-tail labels (double check, though factory filtered it)
            relevant_true_labels = [c for c in true_labels if c in long_tail_codes]

            predicted_labels = classifier.classify(text)

            # Save prediction data
            predictions_data.append(
                {
                    "akte_id": akte_id,
                    "text_snippet": text[:30] + "...",
                    "true_labels": str(relevant_true_labels),
                    "predicted_labels": str(predicted_labels),
                    "all_true_labels": str(true_labels),
                }
            )

            # For evaluation, we need to map these to a consistent binary format or similar
            # We can create binary vectors for the long_tail_codes.

            true_binary = [
                1 if c in relevant_true_labels else 0 for c in long_tail_codes
            ]
            pred_binary = [1 if c in predicted_labels else 0 for c in long_tail_codes]

            all_true_labels.append(true_binary)
            all_pred_labels.append(pred_binary)

            count += 1

        return predictions_data, all_true_labels, all_pred_labels

    def _save_predictions(self, predictions_data):
        """
        Saves predictions to CSV.
        """
        safe_model_name = self.model_name.replace("/", "_")
        if predictions_data:
            df_preds = pd.DataFrame(predictions_data)
            pred_csv_path = f"artifacts/csv/llm_predictions_{safe_model_name}.csv"
            df_preds.to_csv(pred_csv_path, index=False)
            logger.success(f"Saved predictions to {pred_csv_path}")
            self.mlflow_logger.log_artifact(pred_csv_path)

    def _evaluate(self, true_labels, pred_labels, long_tail_codes):
        """
        Calculates metrics and generates plots.
        """
        if not true_labels:
            return

        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)

        safe_model_name = self.model_name.replace("/", "_")

        # Set run name to model name and log tags
        self.mlflow_logger.log_tags(
            {"mlflow.runName": self.model_name, "text_model_name": self.model_name}
        )

        # Calculate scalar metrics using Evaluator
        class_names = [str(c) for c in long_tail_codes]
        evaluator = Evaluator(num_classes=len(long_tail_codes), class_names=class_names)

        metrics = evaluator.compute_metrics(y_true, y_pred)

        logger.info(f"Evaluation Metrics: {metrics}")
        self.mlflow_logger.log_metrics(metrics, step=0)

        # Generate plots using Evaluator
        tags = {"text_model_name": self.model_name}

        # Per-class metrics
        evaluator.save_per_class_metrics(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/csv/per_class_metrics_{safe_model_name}.csv"
        )

        # Overview Metrics Plot
        evaluator.plot_overview_metrics(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/img/overview_metrics_{safe_model_name}.png"
        )

        # ROC and PR curves
        evaluator.plot_roc_curve(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(
            f"artifacts/img/roc_curve_{safe_model_name}.png"
        )

        evaluator.plot_pr_curve(y_true, y_pred, tags=tags)
        self.mlflow_logger.log_artifact(f"artifacts/img/pr_curve_{safe_model_name}.png")

    def run(self):
        logger.info(
            f"Starting LLM classification (threshold={self.threshold}, limit={self.limit})"
        )

        long_tail_codes, descriptions, factory, classifier = self._load_resources()

        if not classifier:
            return

        predictions_data, all_true_labels, all_pred_labels = self._run_inference(
            factory, classifier, long_tail_codes
        )

        self._save_predictions(predictions_data)
        self._evaluate(all_true_labels, all_pred_labels, long_tail_codes)
