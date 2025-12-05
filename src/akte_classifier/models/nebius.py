import os
import time
import torch
import torch.nn as nn
from openai import OpenAI
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

class NebiusTextVectorizer(nn.Module):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.api_key = os.environ.get("NEBIUS_API_KEY")
        
        # We maken hier geen client aan, maar doen dat per thread voor thread-safety
        if not self.api_key:
            raise ValueError("NEBIUS_API_KEY environment variabele is niet ingesteld!")

        # Compatibility fix voor dataset.py
        self.model = nn.Module()
        self.model.dummy_param = nn.Parameter(torch.empty(0))

        # Detecteer hidden size (even 1 snelle check vooraf)
        try:
            client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=self.api_key)
            logger.info(f"Connecting to Nebius API ({model_name})...")
            resp = client.embeddings.create(
                model=model_name,
                input="test",
                encoding_format="float"
            )
            self._hidden_size = len(resp.data[0].embedding)
            logger.success(f"Nebius Connected. Hidden size: {self._hidden_size}")
        except Exception as e:
            logger.error(f"Failed to connect to Nebius: {e}")
            raise e

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _process_batch(self, batch_data):
        """
        Hulpfunctie die door één thread wordt uitgevoerd.
        """
        index, texts = batch_data
        client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=self.api_key)
        
        # Max chars instelling
        MAX_CHARS = 16000
        cleaned_batch = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]

        # Retry logica
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_batch,
                    encoding_format="float"
                )
                # Return tuple (index, embeddings) om volgorde te herstellen
                return (index, [d.embedding for d in response.data])
            except Exception as e:
                if "400" in str(e): 
                    # 400 Bad Request is vaak fataal (te lang), maar we hebben al getruncated.
                    # Soms is het een specifieke corrupte string.
                    logger.error(f"400 Error in thread: {e}. Skipping fallback.")
                    # Return zero vectors als noodoplossing
                    return (index, [[0.0] * self.hidden_size] * len(texts))
                
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt) # Backoff
                else:
                    logger.error(f"Batch failed na retries: {e}")
                    raise e

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.model.dummy_param.device
        if not texts:
            return torch.empty(0, self.hidden_size).to(device)

        if isinstance(texts, tuple): texts = texts[0]
        if isinstance(texts, str): texts = [texts]

        # --- PARALLEL PROCESSING SETTINGS ---
        THREAD_WORKERS = 20   # Aantal gelijktijdige verbindingen
        INTERNAL_BATCH = 16   # Aantal teksten per verbinding

        # 1. Verdeel de input lijst in kleine stukjes (chunks)
        chunks = []
        for i in range(0, len(texts), INTERNAL_BATCH):
            chunk = texts[i : i + INTERNAL_BATCH]
            chunks.append((i, chunk))

        results = []
        
        # 2. Start de ThreadPool
        # Dit vuurt 20 verzoeken tegelijk af op Nebius
        with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
            # Map voert de functie uit voor elke chunk
            processed_chunks = list(executor.map(self._process_batch, chunks))
        
        # 3. Sorteer resultaten (belangrijk! threads finishen willekeurig)
        processed_chunks.sort(key=lambda x: x[0])
        
        # 4. Voeg alles samen
        for _, embeddings in processed_chunks:
            results.extend(embeddings)

        return torch.tensor(results, dtype=torch.float32).to(device)