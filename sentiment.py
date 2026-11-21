"""IMDb sentiment experiment: fixed splits, train-only vocabulary, TextCNN and baselines."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATASET = "stanfordnlp/imdb"
DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
PAD, UNK = 0, 1


@dataclass
class Config:
    seed: int = 42
    vocab_size: int = 20_000
    max_length: int = 256
    embedding_dim: int = 128
    num_filters: int = 100
    dropout: float = 0.5
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def build_vocab(texts: list[str], max_size: int) -> dict[str, int]:
    counts = Counter(word for text in texts for word in text.split())
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for word, _ in counts.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int], length: int) -> torch.Tensor:
    result = torch.zeros((len(texts), length), dtype=torch.long)
    for row, text in enumerate(texts):
        ids = [vocab.get(word, UNK) for word in text.split()[:length]]
        result[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return result


class TextCNN(nn.Module):
    """Embedding -> parallel width 3/4/5 convolutions -> max pooling -> dropout -> logits."""

    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.num_filters, width)
            for width in (3, 4, 5)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(3 * config.num_filters, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_splits(seed: int, smoke: bool = False):
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=DATASET_REVISION)
    labels = np.asarray(dataset["train"]["label"])
    train_ids, val_ids = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=seed, stratify=labels
    )
    test_ids = np.arange(len(dataset["test"]))

    def subset(ids, y, size):
        return train_test_split(ids, train_size=size, stratify=y[ids], random_state=seed)[0]

    if smoke:
        train_ids = subset(train_ids, labels, 128)
        val_ids = subset(val_ids, labels, 64)
        test_ids = subset(test_ids, np.asarray(dataset["test"]["label"]), 64)

    splits = {}
    manifest = {}
    for name, source, ids in [
        ("train", "train", train_ids), ("validation", "train", val_ids),
        ("test", "test", test_ids),
    ]:
        selected = dataset[source].select(ids.tolist())
        splits[name] = ([clean_text(text) for text in selected["text"]], np.asarray(selected["label"]))
        manifest[name] = {
            "source_split": source, "count": len(ids),
            "indices_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        }
    return splits, manifest


def evaluate_predictions(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=["Negative", "Positive"],
            output_dict=True, zero_division=0,
        ),
    }


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = total_correct = count = 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, total_correct / count


@torch.no_grad()
def predict(text: str, model, vocab, config, device):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter a review containing English words or numbers.")
    model.eval()
    logits = model(encode([cleaned], vocab, config.max_length).to(device))
    probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    label = int(np.argmax(probabilities))
    return {
        "label": ["Negative", "Positive"][label],
        "score": probabilities[label],
        "truncated": len(cleaned.split()) > config.max_length,
    }


def save_figures(history, results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    steps = range(1, len(history) + 1)
    for prefix, label in [("train", "Train"), ("val", "Validation")]:
        axes[0].plot(steps, [h[f"{prefix}_loss"] for h in history], marker="o", label=label)
        axes[1].plot(steps, [h[f"{prefix}_accuracy"] for h in history], marker="o", label=label)
    for ax, title in zip(axes, ["Cross-entropy loss", "Accuracy"]):
        ax.set(title=title, xlabel="Epoch", xticks=list(steps))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
    ConfusionMatrixDisplay(np.asarray(results["TextCNN"]["test"]["confusion_matrix"]),
                           display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("TextCNN: held-out IMDb test set")
    fig.savefig(output / "confusion_matrix.svg")
    plt.close(fig)


def train(config: Config, output: Path, device_name="auto", smoke=False):
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = select_device(device_name)
    splits, manifest = load_splits(config.seed, smoke)
    print(f"Device: {device}; splits: { {k: len(v[1]) for k, v in splits.items()} }", flush=True)
    train_texts, train_labels = splits["train"]
    results = {}
    for name, estimator in [("TF-IDF + Naive Bayes", MultinomialNB()),
                            ("TF-IDF + Linear SVM", LinearSVC(random_state=config.seed, max_iter=5000))]:
        pipeline = make_pipeline(TfidfVectorizer(max_features=20_000, ngram_range=(1, 2)), estimator)
        pipeline.fit(train_texts, train_labels)
        results[name] = {split: evaluate_predictions(labels, pipeline.predict(texts))
                         for split, (texts, labels) in splits.items() if split != "train"}
        print(f"{name}: validation accuracy {results[name]['validation']['accuracy']:.4f}", flush=True)

    vocab = build_vocab(train_texts, config.vocab_size)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {name: DataLoader(
        TensorDataset(encode(texts, vocab, config.max_length), torch.tensor(labels, dtype=torch.long)),
        batch_size=config.batch_size, shuffle=name == "train", generator=generator,
    ) for name, (texts, labels) in splits.items()}
    model = TextCNN(len(vocab), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for step in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = epoch(model, loaders["train"], device, optimizer)
        val_loss, val_acc = epoch(model, loaders["validation"], device)
        history.append(dict(epoch=step, train_loss=train_loss, train_accuracy=train_acc,
                            val_loss=val_loss, val_accuracy=val_acc,
                            seconds=time.perf_counter() - started))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {step}/{config.epochs}: train {train_acc:.4f}, validation {val_acc:.4f}; "
              f"{history[-1]['seconds']:.1f}s", flush=True)
    model.load_state_dict(best_state)
    results["TextCNN"] = {}
    for name in ("validation", "test"):
        guesses = []
        model.eval()
        with torch.no_grad():
            for tokens, _ in loaders[name]:
                guesses.extend(model(tokens.to(device)).argmax(1).cpu().tolist())
        results["TextCNN"][name] = evaluate_predictions(splits[name][1], guesses)
    torch.save({"state_dict": best_state, "vocab": vocab, "config": asdict(config)}, output / "textcnn.pt")
    examples = ["This movie was absolutely fantastic!", "I hated this movie. It was boring.",
                "The acting was good, but the story was terrible.", "One of the best films I have ever seen."]
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET, "dataset_revision": DATASET_REVISION, "split_manifest": manifest,
        "config": asdict(config), "smoke_run": smoke, "device": str(device),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "datasets", "numpy", "scikit-learn"]},
        "best_epoch": best_epoch, "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history, "results": results,
        "examples": [{"text": text, **predict(text, model, vocab, config, device)} for text in examples],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    save_figures(history, results, output)
    print(json.dumps({k: v["test"]["accuracy"] for k, v in results.items()}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--output", type=Path, default=Path("runs/full"))
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    training.add_argument("--smoke", action="store_true")
    inference = subparsers.add_parser("predict")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        train(Config(epochs=args.epochs), args.output, args.device, args.smoke)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = Config(**checkpoint["config"])
        model = TextCNN(len(checkpoint["vocab"]), config)
        model.load_state_dict(checkpoint["state_dict"])
        print(json.dumps(predict(args.text, model, checkpoint["vocab"], config, torch.device("cpu")), indent=2))


if __name__ == "__main__":
    main()
