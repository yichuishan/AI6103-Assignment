# Assignment contract

The uploaded `AI6103_Assignment.pdf` is the requirements source. It allows
generative-AI assistance but makes the student responsible for checking the report.
No answer to the assignment's prohibited “Takeaway Message” question is included.

| Section | Conditions | Epochs each | Fixed settings |
|---|---|---:|---|
| Preprocessing | 7,155 train / 1,533 validation / 1,534 test, 120 breeds | — | 100×100; training-only RGB population mean/std |
| Initial LR | 0.5, 0.005, 0.001 | 15 | SGD, momentum 0.9, batch 128, decay 0, constant LR |
| Schedule | constant, cosine ending at zero | 300 | LR minimizing final training CE; decay 0 |
| Weight decay | 0.0005, 0.0001 | 300 | same LR, cosine; reuse zero-decay cosine control |
| Mixup | alpha 0.2 | 300 | best pre-mixup long configuration by final validation accuracy |
| Final test | selected final checkpoint | once | validation-only selection; no test-guided tuning |

Every model is torchvision EfficientNetV2-S (`weights=None`, default SiLU),
reinitialized with the same seed. There are **8 runs, 1,545 total epochs**.
Cropping uses 12-pixel black padding then a 100×100 crop; horizontal flip
probability is 0.5. No pretraining, alternative optimizer, ImageNet normalization,
early stopping, or reduced full-run epoch budgets.

## Measurement conventions

- `train_loss`, `train_accuracy`: full clean training-set evaluation after each
  epoch, in eval mode, without augmentation. `val_*` uses the same deterministic
  evaluation transform. These define the reported tables and required curves.
- `optimization_loss`, `optimization_accuracy`: online augmented training metrics.
  Mixup accuracy is lambda-weighted correctness against both targets; its CE is
  the lambda-weighted target CE. This is not hard-label accuracy on dogs.
- CE excludes the weight penalty; metrics average samples, not batches.
  Accuracy is stored as a fraction and plotted/reported as a percent.
- Initial-LR selection minimizes **final clean training CE**, as prescribed;
  ties use lower LR. Other choices maximize **final validation accuracy**, then
  minimize final validation CE, then use lexical run name to break exact ties.
- Final selection includes all five 300-epoch runs, using `latest.pt`.
  `best.pt` is retained for inspection, never silently used for final metrics.
- SGD uses coupled weight decay on all parameters. With momentum this differs
  from explicitly decoupled decay. Cosine steps after each epoch: epoch 1 uses
  eta_0, and the stored rate reaches zero after epoch 300. No warm restarts.

## Data and reproducibility

Provided splits take priority. A CSV manifest has `path,label,split` columns,
paths absolute or relative to `--data-root`, splits exactly `train,val,test`.
Counts, known labels, duplicate paths and cross-split byte duplicates are checked.
For supplied splits, same-label byte duplicates are audited and disclosed while
preserving the assignment's exact split membership and counts. Their overlap can
bias validation/test scores upward, so those estimates must not be described as
fully independent unseen-image performance. Conflicting labels for identical bytes
still fail validation. Newly generated splits retain the stricter no-overlap check.
The audit is stored in prepared data and `results/data_quality_audit.json`, copied
into report inserts and the final results package, and stated in the abstract,
preprocessing, final evaluation and limitations whenever overlap is present.
Hash checking reads image bytes but never evaluates test predictions for tuning.
Only training pixels contribute to normalization statistics.

If official splits are unavailable, `--allow-resplit` explicitly enables an
exact-size stratified split of `labels.csv` plus `train/<id>.jpg`. Do not call
that an official provided split. It is saved in `prepared.json`; training never
regenerates it. Near-duplicate photos are not detected. Paths are fingerprinted:
moving data requires a new prepared file and fresh run directory.

Each run records config, actual versions/device, histories, latest/best checkpoints
and RNG state. Repeating `run` resumes the last complete epoch; config/data/seed
mismatches are rejected. Exact resumption was tested on the same CPU runtime with
zero workers. Cross-hardware bitwise reproducibility is not assumed.

`test` requires selection saved only after all eight runs finish and caches the
evaluation. Changing selection against an existing test record is rejected.
Do not delete test records to retune on hold-out data. Load only your own trusted
checkpoints: Python RNG state requires `torch.load(weights_only=False)`.

## Artifacts and report

`runs/full/<name>/`: config, history CSV, latest/best checkpoints and summary.
Global selection/test records are in `runs/full/`. `plot` creates four measured
training-curve comparisons plus beta/cosine illustrations as PDF and PNG.
`report` creates measured CSV/LaTeX tables, rankings, numerical comparisons,
trajectory-based observations, abstract, and runtime limitations; smoke results
are rejected. The English report uses unmodified AAAI 2024 style and includes
preprocessing code/equations. It stays explicitly incomplete until the real
experiments and hold-out evaluation exist. Read the generated analysis and
check the figures before submitting; single-seed outcomes are not significance tests.

## One-command AutoDL execution

`bash run_autodl.sh /root/autodl-tmp/data.zip` checks dependencies and TeX,
extracts ZIP safely into `results/extracted/<archive hash>`, discovers supported
labelled layouts, measures training-only statistics, checks an actual GPU batch
of 128, validates a draft PDF, and runs tests before launching background training.
Existing CUDA PyTorch is retained; missing CUDA wheels use the official cu124
index. Other Python dependencies are pinned by `requirements.txt`. Missing TeX
tools are installed using apt on root Debian/Ubuntu hosts. Each installation
retains normal TLS/package verification. Installation failures stop before training.

Automatic precision is BF16 on a supporting GPU (including supported Ampere
configurations), otherwise FP16 with gradient scaling. The actual setting is
recorded and shared by all conditions. Evaluation uses FP32, without autocast.
`--precision fp32` forces full precision if desired. Batch size remains 128;
the script never substitutes microbatches or changes the epoch budgets.

All generated files, logs, inputs, results and models stay under `results/`.
The default detached process survives closing the terminal and holds an exclusive
workflow lock; identical invocation resumes after an interruption. Worker count,
seed, precision and data-source mismatches require a new output directory.
`results/status.json` distinguishes preflight, running, failed and complete.
The final `AI6103_results.zip` includes report sources/PDF, figures, numeric tables,
histories and selection records. Large model checkpoints remain under `results/runs/`
and are intentionally excluded from this portable report package.

The full-report layout is tested with artificial artifact fixtures kept outside
the repository. Such fixtures verify rendering only, never empirical performance.
Full GPU training remains a runtime operation on the user's AutoDL server.
