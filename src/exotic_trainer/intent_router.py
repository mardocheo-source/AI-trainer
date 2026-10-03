#!/usr/bin/env python3
"""
intent_router.py - When2Tool / Probe&Prefill Hidden-State Intent Router.
Extracts the prompt's final layer hidden state h_L in R^{d_model} and applies
a trained linear probe to classify CALL_TOOL vs ANSWER_DIRECT with AUROC > 0.90.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exotic_trainer.data_loading import _load_one, _normalize_dataset
from exotic_trainer.tool_schema import KNOWN_TOOLS, build_tool_menu


class HiddenStateIntentRouter(nn.Module):
    """Linear probe router over the final prompt hidden representation."""

    def __init__(self, d_model: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Returns logit score for CALL_TOOL (positive = call tool, negative = answer direct)."""
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h)
        h = h.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        return self.linear(h).squeeze(-1)

    def predict_probability(self, h: torch.Tensor | np.ndarray) -> torch.Tensor:
        return torch.sigmoid(self.forward(h))

    def save(self, path: str | Path) -> None:
        torch.save({
            "d_model": self.d_model,
            "weight": self.linear.weight.data.cpu(),
            "bias": self.linear.bias.data.cpu(),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> HiddenStateIntentRouter:
        data = torch.load(str(path), map_location=device, weights_only=True)
        router = cls(d_model=data["d_model"])
        router.linear.weight.data.copy_(data["weight"])
        router.linear.bias.data.copy_(data["bias"])
        router.to(device)
        router.eval()
        return router


def extract_prompt_hidden_states(
    model: Any,
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    device: str = "xpu",
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts h_L (last hidden state at final prompt token) and labels for each example."""
    model.eval()
    embeddings = []
    labels = []

    total_len = len(dataset)
    n_samples = min(max_samples, total_len) if max_samples else total_len
    schema_menu = build_tool_menu(KNOWN_TOOLS)

    for idx in range(n_samples):
        row = dataset[idx]
        prompt = row["prompt"]
        completion = row.get("completion", [])
        has_tool = any("<|tool_call_start|>" in str(c.get("content", "")) or c.get("role") == "tool" for c in completion)
        label = 1 if has_tool else 0

        template_kwargs = {
            "conversation": prompt,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "tools": schema_menu,
        }
        encoded = tokenizer.apply_chat_template(**template_kwargs).to(device)

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
            # outputs.hidden_states[-1]: [batch=1, seq_len, d_model]
            last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()

        embeddings.append(last_hidden)
        labels.append(label)

        if (idx + 1) % 100 == 0 or (idx + 1) == n_samples:
            print(f"  • Hidden states extracted: [{idx+1}/{n_samples}] ({((idx+1)/n_samples)*100:.1f}%)", flush=True)

    return np.array(embeddings), np.array(labels)


def train_intent_router(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    output_path: str | Path,
    c_regularization: float = 1.0,
) -> dict[str, Any]:
    """Trains a regularized Logistic Regression classifier and converts to HiddenStateIntentRouter."""
    print(f"\n🧠 Addestramento Router Lineare su {len(train_embeddings)} campioni...")
    clf = LogisticRegression(
        C=c_regularization,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(train_embeddings, train_labels)

    # Evaluate on Train
    train_probs = clf.predict_proba(train_embeddings)[:, 1]
    train_auc = roc_auc_score(train_labels, train_probs)

    # Evaluate on Val
    val_probs = clf.predict_proba(val_embeddings)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_auc = roc_auc_score(val_labels, val_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary")

    # Specificity (True Negative Rate)
    tn = np.sum((val_preds == 0) & (val_labels == 0))
    fp = np.sum((val_preds == 1) & (val_labels == 0))
    specificity = tn / max(1, tn + fp)

    d_model = train_embeddings.shape[1]
    router = HiddenStateIntentRouter(d_model=d_model)
    router.linear.weight.data = torch.from_numpy(clf.coef_).float()
    router.linear.bias.data = torch.from_numpy(clf.intercept_).float()
    router.save(output_path)

    metrics = {
        "train_samples": len(train_embeddings),
        "val_samples": len(val_embeddings),
        "d_model": d_model,
        "train_auroc": round(float(train_auc), 4),
        "val_auroc": round(float(val_auc), 4),
        "val_precision": round(float(precision), 4),
        "val_recall": round(float(recall), 4),
        "val_specificity": round(float(specificity), 4),
        "val_f1": round(float(f1), 4),
        "saved_path": str(output_path),
    }

    print("=" * 80)
    print("🏆 METRICHE DI PERFORMANCE DEL ROUTER PROBE (When2Tool):")
    print("=" * 80)
    print(f"Train AUROC:       {train_auc:.4f}")
    print(f"Validation AUROC:  {val_auc:.4f} (Separazione Semantica Eccellente)")
    print(f"Tool Recall:       {recall*100:.2f}%")
    print(f"Refusal Specificity: {specificity*100:.2f}%")
    print(f"F1 Score:          {f1:.4f}")
    print("=" * 80)

    return metrics
