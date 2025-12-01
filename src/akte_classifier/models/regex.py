import csv
import re
from typing import Dict, List

import torch
from loguru import logger


class RegexGenerator:
    """
    Based on a CSV file with code and description,
    this class automatically generate regex patterns for each code.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.regexes: Dict[int, str] = {}
        self._generate_regexes()

    def _generate_regexes(self):
        """
        Parses the CSV and generates regex patterns for each code.
        """
        logger.info(f"Parsing {self.csv_path} for regex generation...")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            # The CSV uses ';' as delimiter based on user input
            reader = csv.reader(f, delimiter=";")

            # Skip header if present (Code;Waarde)
            header = next(reader, None)
            if header and header[0].lower() != "code":
                # If first row doesn't look like header, reset
                f.seek(0)

            for row in reader:
                if not row or len(row) < 2:
                    continue

                try:
                    code = int(row[0].strip())
                    description = row[1].strip()

                    # Generate regex from description
                    pattern = self._create_pattern(description)
                    self.regexes[code] = pattern

                except ValueError:
                    continue

        logger.info(f"Generated {len(self.regexes)} regex patterns.")

    def generate_hash(self) -> str:
        """
        Generates a hash so we can spot changes in the regex patterns.
        This helps with caching
        """
        import hashlib

        # Sort by code to ensure stability
        sorted_items = sorted(self.regexes.items())
        # Create a string representation
        repr_str = str(sorted_items)
        # Hash it
        return hashlib.md5(repr_str.encode()).hexdigest()[:8]

    def _create_pattern(self, description: str) -> str:
        """
        Converts a description into a flexible regex pattern.
        """
        # Handle "enz." (etc.) - remove it
        clean_desc = description.replace("enz.", "").strip()

        # Logic: If parentheses exist,
        # eg mandeligheid (wijziging)
        # we treat the parts inside and outside as separate required components.
        # We assume the order in the description matters
        # So we use "part1.*part2" which is much faster than lookaheads.
        if "(" in clean_desc and ")" in clean_desc:
            # Split by parens
            parts = re.split(r"[\(\)]", clean_desc)
            # Filter empty and strip
            parts = [p.strip() for p in parts if p.strip()]

            pattern_parts = []
            for p in parts:
                escaped = re.escape(p)
                # Flexible whitespace
                escaped = escaped.replace(r"\ ", r"\s{1,5}")
                pattern_parts.append(escaped)

            # Combine with .* to match in order, allowing anything in between
            pattern = ".*".join(pattern_parts)
        else:
            # Standard handling
            escaped = re.escape(clean_desc)
            # Flexible whitespace
            pattern = escaped.replace(r"\ ", r"\s{1,5}")

        return pattern


class RegexClassifier:
    def __init__(self, generator: RegexGenerator):
        self.patterns = {
            code: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for code, pattern in generator.regexes.items()
        }

    def predict(self, text: str) -> List[int]:
        """
        Returns a list of codes found in the text.
        """
        matches = []
        for code, pattern in self.patterns.items():
            if pattern.search(text):
                matches.append(code)
        return matches


class RegexVectorizer:
    def __init__(self, generator: RegexGenerator, label_encoder=None):
        self.generator = generator
        self.hash = generator.generate_hash()

        """
        Args:
            generator: The RegexGenerator instance.
            label_encoder: Optional LabelEncoder to align outputs with model classes.
        """
        self.output_dim = 0
        self.code_to_idx = {}
        # Use the simple regexes (code -> pattern)
        self.patterns = {}

        if label_encoder is None:
            logger.error(
                "LabelEncoder is required for RegexVectorizer to align with model classes."
            )
            raise ValueError("LabelEncoder is required.")

        # Single Source of Truth: Use the encoder's codes and indices
        self.code_to_idx = label_encoder.code2idx
        self.output_dim = len(label_encoder)

        # Only compile patterns for codes that are in the encoder (and have a regex)
        for code_str, idx in self.code_to_idx.items():
            try:
                code_int = int(code_str)
                if code_int in generator.regexes:
                    self.patterns[idx] = re.compile(
                        generator.regexes[code_int], re.IGNORECASE | re.DOTALL
                    )
            except ValueError:
                continue

    def _match_text(self, text: str) -> List[int]:
        """Helper for parallel processing: returns list of indices that matched."""
        matches = []
        for idx, pattern in self.patterns.items():
            if pattern.search(text):
                matches.append(idx)
        return matches

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Transforms a list of texts into a binary tensor of shape (batch_size, num_classes).
        Values:
        0: No match
        1: Match
        """
        from joblib import Parallel, delayed

        batch_size = len(texts)
        features = torch.zeros(
            (batch_size, self.output_dim), dtype=torch.float32
        )

        # Parallelize the matching process
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=-1)(
            delayed(self._match_text)(text) for text in texts
        )

        for i, matched_indices in enumerate(results):
            if matched_indices:
                features[i, matched_indices] = 1.0

        return features
