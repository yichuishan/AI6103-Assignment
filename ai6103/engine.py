"""Training, resumable SGD experiments, validation-only selection and final test."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as vision_version
from torchvision.models import efficientnet_v2_s

from .data import CLEANING_POLICY, DogDataset, load_json, save_json


@dataclass(frozen=True)
class Experiment:
    name: str
    stage: str
    lr: float
    epochs: int
    schedule: str = "constant"
    weight_decay: float = 0.
    mixup_alpha: float = 0.
    batch_size: int = 128
    seed: int = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def make_model(classes):
    # No pretraining; keep torchvision's default SiLU and architecture.
    return efficientnet_v2_s(weights=None, num_classes=classes)


def loader(prepared, split, batch_size, workers, generator=None, augment=False, device="cpu"):
    return DataLoader(DogDataset(prepared, split, augment), batch_size=batch_size,
                      shuffle=augment, drop_last=False, num_workers=workers,
                      pin_memory=str(device).startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def mix_batch(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[permutation], y, y[permutation], lam


def train_epoch(model, batches, optimizer, device, alpha=0., precision="fp32", scaler=None):
    model.train()
    total_loss = total_correct = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=str(device).split(":")[0],
                            dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                            enabled=precision != "fp32"):
            if alpha:
                x, ya, yb, lam = mix_batch(x, y, alpha)
                logits = model(x)
                loss = lam * nn.functional.cross_entropy(logits, ya) + (1-lam) * nn.functional.cross_entropy(logits, yb)
                correct = lam * (logits.argmax(1) == ya).sum().item() + (1-lam) * (logits.argmax(1) == yb).sum().item()
            else:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                correct = (logits.argmax(1) == y).sum().item()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite optimization loss; this run is failed, not completed")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        total_correct += correct
        total += len(y)
    if not total:
        raise ValueError("Empty training loader")
    return total_loss / total, total_correct / total


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    total_loss = total_correct = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    if not total:
        raise ValueError("Empty evaluation loader")
    return {"loss": total_loss / total, "accuracy": total_correct / total, "n": total}


def atomic_checkpoint(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_history(path, history):
    if not history:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(stream.getvalue())
    tmp.replace(path)


def checkpoint_load(path):
    # Checkpoints contain Python/NumPy RNG state. Load only your own trusted artifacts.
    return torch.load(path, map_location="cpu", weights_only=False)


def run_experiment(prepared, exp, out, device="cpu", workers=0, smoke=False, stop_after=None, precision="fp32"):
    out = Path(out) / exp.name
    out.mkdir(parents=True, exist_ok=True)
    config = {"experiment": asdict(exp), "data_fingerprint": prepared["fingerprint"],
              "precision": precision,
              "smoke": smoke, "metric_definition": "train/val are clean unaugmented eval-mode CE/top1; optimization metrics use augmentation/mixup"}
    config_path = out / "config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(f"Configuration mismatch for {out}; choose another output directory")
    save_json(config_path, config)
    if (out / "summary.json").exists():
        return load_json(out / "summary.json")
    seed_all(exp.seed)
    generator = torch.Generator().manual_seed(exp.seed)
    model = make_model(len(prepared["classes"])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=exp.lr, momentum=0.9,
                                weight_decay=exp.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp.epochs, eta_min=0.)
                 if exp.schedule == "cosine" else None)
    train_batches = loader(prepared, "train", exp.batch_size, workers, generator, True, device)
    clean_batches = loader(prepared, "train", exp.batch_size, workers, device=device)
    val_batches = loader(prepared, "val", exp.batch_size, workers, device=device)
    latest = out / "latest.pt"
    history, best_acc = [], -1.
    if latest.exists():
        state = checkpoint_load(latest)
        if state["config"] != config:
            raise ValueError("Checkpoint config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        history, best_acc = state["history"], state["best_acc"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if str(device).startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    write_history(out / "history.csv", history)
    for epoch in range(len(history), exp.epochs):
        if stop_after is not None and epoch >= stop_after:
            return {"status": "interrupted", "epochs_completed": len(history)}
        started = time.perf_counter()
        lr_used = optimizer.param_groups[0]["lr"]
        optimization_loss, optimization_accuracy = train_epoch(model, train_batches, optimizer, device,
                                                               exp.mixup_alpha, precision, scaler)
        train_metrics = evaluate(model, clean_batches, device)
        val_metrics = evaluate(model, val_batches, device)
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch+1, "lr": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
               "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
               "optimization_loss": optimization_loss, "optimization_accuracy": optimization_accuracy,
               "seconds": time.perf_counter()-started}
        history.append(row)
        is_best = row["val_accuracy"] > best_acc
        best_acc = max(best_acc, row["val_accuracy"])
        if is_best:
            atomic_checkpoint(out / "best.pt", {"model": model.state_dict(), "epoch": epoch+1,
                                                "config": config, "val_accuracy": best_acc})
        state = {"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "history": history, "best_acc": best_acc,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None}
        atomic_checkpoint(latest, state)
        write_history(out / "history.csv", history)
        print(f"{exp.name} {epoch+1}/{exp.epochs}: train={row['train_loss']:.4f}/{row['train_accuracy']:.3%} "
              f"val={row['val_loss']:.4f}/{row['val_accuracy']:.3%} ({row['seconds']:.1f}s)", flush=True)
    summary = {"status": "complete", "config": config, "final": history[-1],
               "best_val_accuracy": best_acc,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "torchvision": vision_version, "device": str(device),
                           "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}}
    save_json(out / "summary.json", summary)
    return summary


def lr_experiments(seed=42, smoke=False):
    # The tiny fixture's batch-2 gradients are unstable at the assignment's 0.5.
    # Smoke checks control flow only; formal rates and epoch budgets stay fixed.
    rates = (0.0005, 0.00005, 0.00001) if smoke else (0.5, 0.005, 0.001)
    return [Experiment(f"lr_{lr:g}", "lr", lr, 2 if smoke else 15,
                       batch_size=2 if smoke else 128, seed=seed) for lr in rates]


def completed(root, names):
    results = []
    for name in names:
        p = Path(root) / name / "summary.json"
        if not p.exists():
            raise ValueError(f"Prerequisite {name} is incomplete")
        result = load_json(p)
        if result["status"] != "complete":
            raise ValueError(f"Prerequisite {name} failed")
        results.append(result)
    return results


def best_validation(results):
    return min(results, key=lambda r: (-r["final"]["val_accuracy"], r["final"]["val_loss"],
                                       r["config"]["experiment"]["name"]))


def run_pipeline(prepared, root, stage="all", device="cpu", workers=0, seed=42, smoke=False, precision="fp32"):
    if not smoke and not prepared.get("assignment_data"):
        raise ValueError("Non-assignment data cannot be used for full experiments")
    if not smoke and prepared.get("cleaning",{}).get("policy") != CLEANING_POLICY:
        raise ValueError("Prepared data use the old preprocessing policy; prepare cleaned data and use a new run directory")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    binding = {"data_fingerprint": prepared["fingerprint"], "seed": seed, "smoke": smoke, "precision": precision}
    if (root / "pipeline.json").exists() and load_json(root / "pipeline.json") != binding:
        raise ValueError("Run directory belongs to another dataset/seed/mode")
    save_json(root / "pipeline.json", binding)
    stages = ["lr", "schedule", "decay", "mixup"] if stage == "all" else [stage]
    for current in stages:
        if current == "lr":
            experiments = lr_experiments(seed, smoke)
        else:
            first = completed(root, [e.name for e in lr_experiments(seed, smoke)])
            # The assignment explicitly asks for the LR minimizing final training loss.
            best_lr_result = min(first, key=lambda r: (r["final"]["train_loss"], r["config"]["experiment"]["lr"]))
            best_lr = best_lr_result["config"]["experiment"]["lr"]
            common = dict(lr=best_lr, epochs=2 if smoke else 300,
                          batch_size=2 if smoke else 128, seed=seed)
            if current == "schedule":
                experiments = [Experiment("schedule_constant", current, schedule="constant", **common),
                               Experiment("schedule_cosine", current, schedule="cosine", **common)]
            elif current == "decay":
                completed(root, ["schedule_constant", "schedule_cosine"])
                experiments = [Experiment(f"decay_{wd:g}", current, schedule="cosine", weight_decay=wd, **common)
                               for wd in (5e-4, 1e-4)]
            else:
                candidates = completed(root, ["schedule_constant", "schedule_cosine", "decay_0.0005", "decay_0.0001"])
                base = best_validation(candidates)
                base_exp = base["config"]["experiment"]
                save_json(root / "mixup_baseline.json", {"name": base_exp["name"],
                          "reason": "maximum final validation accuracy; ties: lower validation CE, then name"})
                experiments = [Experiment("mixup", current, schedule=base_exp["schedule"],
                                          weight_decay=base_exp["weight_decay"], mixup_alpha=0.2, **common)]
        for exp in experiments:
            run_experiment(prepared, exp, root, device, workers, smoke, precision=precision)
    if stage in ("all", "mixup"):
        candidates = completed(root, [e.name for e in lr_experiments(seed, smoke)] +
                               ["schedule_constant", "schedule_cosine", "decay_0.0005", "decay_0.0001", "mixup"])
        long_runs = [r for r in candidates if r["config"]["experiment"]["stage"] != "lr"]
        winner = best_validation(long_runs)
        save_json(root / "selection.json", {
            "name": winner["config"]["experiment"]["name"], "checkpoint": "latest.pt",
            "rule": "maximum final validation accuracy across 300-epoch runs; ties lower CE, then name",
            "data_fingerprint": prepared["fingerprint"], "smoke": smoke,
            "validation_accuracy": winner["final"]["val_accuracy"]})


def test_once(prepared, root, device="cpu", workers=0):
    root = Path(root)
    selection = load_json(root / "selection.json")
    if selection["data_fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Selection dataset mismatch")
    signature = hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
    target = root / "test_result.json"
    if target.exists():
        result = load_json(target)
        if result["selection_sha256"] != signature:
            raise ValueError("Test was already used with a different selection; do not tune on test")
        return result
    state = checkpoint_load(root / selection["name"] / selection["checkpoint"])
    if state["config"]["data_fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Test checkpoint data mismatch")
    model = make_model(len(prepared["classes"])).to(device)
    model.load_state_dict(state["model"])
    metrics = evaluate(model, loader(prepared, "test", 128, workers, device=device), device)
    result = {"selection_sha256": signature, "name": selection["name"], "metrics": metrics,
              "smoke": selection["smoke"], "checkpoint_epoch": len(state["history"])}
    save_json(target, result)
    return result
