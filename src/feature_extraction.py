"""
CodeBERT-based embedding extraction.

This mirrors the approach used in FlakyQ (Rahman & Misailovic, ICST 2024) and
Flakify (Haben et al.): feed the raw test source code into a pretrained
code-understanding model (CodeBERT) and take a pooled representation of its
final hidden layer as a dense "embedding" for the test.

We do NOT fine-tune CodeBERT in this starter script (fine-tuning needs a
GPU and a lot more data to be worth it). Using the pretrained model directly
("feature extraction mode") is a reasonable, much faster starting point for
a semester project, and you can mention fine-tuning as a future-work item
that would likely improve accuracy further, consistent with what FlakyQ
reports.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

_MODEL_NAME = "microsoft/codebert-base"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading {_MODEL_NAME} (first call only, this can take a minute)...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def get_codebert_embedding(code: str, max_length: int = 256) -> np.ndarray:
    """
    Returns a 768-dimensional numpy vector representing the given test's
    source code, using mean-pooling over CodeBERT's last hidden layer.
    """
    tokenizer, model = _load_model()
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean-pool over the token dimension, ignoring padding tokens.
    last_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
    mask = inputs["attention_mask"].squeeze(0).unsqueeze(-1)  # (seq_len, 1)
    summed = (last_hidden * mask).sum(dim=0)
    counted = mask.sum(dim=0).clamp(min=1)
    embedding = (summed / counted).numpy()
    return embedding


def get_codebert_embeddings_batch(code_list, max_length: int = 256) -> np.ndarray:
    """Convenience wrapper: embeds a list of code strings, one row each."""
    return np.vstack([get_codebert_embedding(c, max_length) for c in code_list])


if __name__ == "__main__":
    sample = "public void testAdd() { assertEquals(2, 1 + 1); }"
    emb = get_codebert_embedding(sample)
    print("Embedding shape:", emb.shape)
    print("First 5 values:", emb[:5])
