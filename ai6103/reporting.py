"""Generate report inserts exclusively from actual, non-smoke run artifacts."""
import csv
import io
import hashlib
from pathlib import Path

from .data import load_json, save_json

EXPECTED = ["lr_0.5", "lr_0.005", "lr_0.001", "schedule_constant", "schedule_cosine",
            "decay_0.0005", "decay_0.0001", "mixup"]


def tex(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(text))


def loss_number(value):
    if abs(value) >= 10000:
        mantissa,exponent=f"{value:.3e}".split("e")
        return f"${mantissa}\\times 10^{{{int(exponent)}}}$"
    return f"{value:.4f}"


def quality_disclosure(prepared):
    audit=(prepared or {}).get("data_quality_audit",{})
    cleaning=(prepared or {}).get("cleaning")
    if cleaning:
        return (
            f"File-content checks found {audit['total_duplicate_groups']} duplicate groups, "
            f"including {audit['cross_split_duplicate_groups']} across splits and "
            f"{audit['conflicting_label_groups']} with conflicting labels. "
            "All records in conflicting-label groups were excluded. For each remaining duplicate group, "
            "one copy was retained, prioritizing test, then validation, then training membership. "
            f"This removed {cleaning['removed_conflict_images']} conflicting-label records and "
            f"{cleaning['removed_duplicate_images']} duplicate copies. "
            "No byte-identical images remain in the retained data. "
        )
    text=""
    if audit.get("cross_split_duplicate_groups",0):
        overlap=audit["overlap_images"]
        text+=(f"A SHA-256 audit found {audit['cross_split_duplicate_groups']} byte-identical cross-split image groups: "
            f"{overlap['train_val']} validation images duplicate training bytes, "
            f"{overlap['train_test']} test images duplicate training bytes, and "
            f"{overlap['val_test']} test images duplicate validation bytes. "
            "This content overlap can bias validation or test performance; these scores are not fully independent estimates on unseen images. ")
    if audit.get("conflicting_label_groups",0):
        text+=(f"The audit also found {audit['conflicting_label_groups']} identical-content groups with inconsistent labels, "
               f"involving {audit['images_in_conflicting_label_groups']} images "
               f"({audit['training_images_in_conflicting_label_groups']} training images). "
               "These contradictory annotations introduce label noise and can limit attainable training accuracy; "
               "no label is chosen as correct without external review. ")
    if text:
        text+="The provided split membership, labels, and required image counts are preserved. "
    return text


def observations(results, names):
    group = [results[n] for n in names if n in results]
    if len(group) != len(names):
        return ("This comparison has not been completed on the assignment dataset. "
                "Final metrics and trajectory-based conclusions remain unreported.\n")
    phrases = []
    for metric, label, reverse in [("train_loss", "lowest training CE", False),
                                   ("train_accuracy", "highest training accuracy", True),
                                   ("val_loss", "lowest validation CE", False),
                                   ("val_accuracy", "highest validation accuracy", True)]:
        target = (max if reverse else min)(r["final"][metric] for r in group)
        winners = [r["config"]["experiment"]["name"] for r in group if r["final"][metric] == target]
        value = f"{100*target:.2f}\\%" if "accuracy" in metric else f"{target:.4f}"
        phrases.append(f"The {label} is {value}, attained by " + ", ".join(r"\texttt{"+tex(n)+"}" for n in winners) + ".")
    return " ".join(phrases) + "\n"


def measured_interpretation(stage, results, names, root):
    if not all(n in results for n in names):
        return ""
    group=[results[n] for n in names]
    if stage == "lr":
        train=min(group,key=lambda r:r["final"]["train_loss"])
        valid=min(group,key=lambda r:(-r["final"]["val_accuracy"],r["final"]["val_loss"]))
        if train is not valid:
            text="The training-loss and validation-accuracy choices differ. Better fitting of the training labels therefore does not translate directly into better held-out classification in these runs. "
        else:
            text="The same condition minimizes training CE and maximizes validation accuracy in this comparison. This agreement is specific to this split and epoch budget. "
    else:
        base=group[0]["final"]
        candidate=(group[1] if stage in {"schedule","mixup"} else
                   min(group[1:],key=lambda r:(-r["final"]["val_accuracy"],r["final"]["val_loss"])))
        end=candidate["final"]
        delta=100*(end["val_accuracy"]-base["val_accuracy"])
        ce_delta=end["val_loss"]-base["val_loss"]
        direction="increased" if delta>0 else "decreased" if delta<0 else "was unchanged"
        text=("Relative to the control, validation accuracy "+direction+
              (f" by {abs(delta):.2f} percentage points" if delta else "")+
              f", while validation CE changed by {ce_delta:+.4f}. ")
        if stage == "schedule":
            if end["train_loss"] < base["train_loss"]:
                text+="The cosine run also has lower training CE, consistent with more precise late-stage optimization. "
            else:
                text+="Cosine annealing does not lower the final training CE here, so these runs do not support an optimization advantage under the chosen schedule. "
        elif delta>0 and end["train_loss"]>base["train_loss"]:
            text+="Higher clean training CE together with improved validation accuracy is consistent with a regularization benefit at the cost of training fit. "
        elif delta<=0:
            text+="This comparison does not show a validation-accuracy benefit from the intervention, even if another metric improves. "
    winner=min(group,key=lambda r:(-r["final"]["val_accuracy"],r["final"]["val_loss"]))
    name=winner["config"]["experiment"]["name"]
    history=Path(root)/name/"history.csv"
    if history.exists():
        with history.open(newline="") as f:
            rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        minimum=min(rows,key=lambda r:r["val_loss"])
        gap=100*(rows[-1]["train_accuracy"]-rows[-1]["val_accuracy"])
        text+=f"For the validation-accuracy winner, the final training--validation accuracy gap is {gap:.2f} percentage points. "
        if minimum["val_loss"]+1e-6<rows[-1]["val_loss"] and rows[-1]["train_loss"]<rows[0]["train_loss"]:
            text+=(f"Its validation CE is lowest at epoch {int(minimum['epoch'])}, then rises to the final value while training CE decreases over the run; this is consistent with late overfitting, although one seed does not establish its cause. ")
    return text+"\n"


def generate(prepared_path, root, output, author="AI6103 Assignment"):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prepared = load_json(prepared_path) if Path(prepared_path).exists() else None
    if prepared and not prepared.get("assignment_data"):
        raise ValueError("Refusing to put synthetic data statistics in the assignment report")
    quality_text=quality_disclosure(prepared)
    audit=(prepared or {}).get("cleaning",{}).get("after",(prepared or {}).get("data_quality_audit",{}))
    has_overlap=audit.get("cross_split_duplicate_groups",0)>0
    has_conflicts=audit.get("conflicting_label_groups",0)>0
    if prepared and "data_quality_audit" in prepared:
        save_json(output/"data_quality_audit.json",prepared["data_quality_audit"])
    results = {}
    for name in EXPECTED:
        p = root / name / "summary.json"
        if not p.exists():
            continue
        result = load_json(p)
        cfg = result["config"]
        if cfg["smoke"]:
            raise ValueError("Refusing to put smoke-test results in the assignment report")
        if prepared is None or cfg["data_fingerprint"] != prepared["fingerprint"]:
            raise ValueError("Report prepared data do not match experiment artifacts")
        expected_epochs = 15 if name.startswith("lr_") else 300
        if result["status"] != "complete" or result["final"]["epoch"] != expected_epochs:
            raise ValueError(f"{name} does not contain all required epochs")
        results[name] = result
    all_complete = len(results) == len(EXPECTED)
    test = load_json(root / "test_result.json") if (root / "test_result.json").exists() else None
    if test and (test["smoke"] or not all_complete):
        raise ValueError("Invalid or premature test result for assignment report")
    if test:
        selection_path=root/"selection.json"
        selection=load_json(selection_path)
        if (test["selection_sha256"] != hashlib.sha256(selection_path.read_bytes()).hexdigest()
            or test["name"] != selection["name"] or selection["data_fingerprint"] != prepared["fingerprint"]):
            raise ValueError("Test result does not match the frozen validation selection")
    status = {"complete_experiments": len(results), "required_experiments": 8,
              "data_available": prepared is not None, "test_evaluated": test is not None,
              "report_complete": all_complete and test is not None,
              "missing_experiments": [n for n in EXPECTED if n not in results]}
    save_json(output / "status.json", status)
    if status["report_complete"]:
        notice = "All eight required experiments and the final hold-out evaluation are recorded."
    else:
        notice = ("INCOMPLETE EXPERIMENTAL DRAFT. "
                  f"{len(results)}/8 required experiments are complete. "
                  "Missing values are not estimates. Synthetic verification results are excluded.")
    (output / "status.tex").write_text((r"\newif\ifExperimentsComplete"+"\n"+
               (r"\ExperimentsCompletetrue" if status["report_complete"] else r"\ExperimentsCompletefalse")+
               "\n"+("" if status["report_complete"] else r"\noindent\textbf{"+tex(notice)+"}\n")))
    (output/"author.tex").write_text(tex(author))
    abstract=("We compare initial learning rate, cosine annealing, weight decay, and mixup "
              "for torchvision EfficientNetV2-S trained from random initialization on 120 dog breeds. "
              "The protocol uses 100-by-100 images, SGD with momentum 0.9, batch size 128, "
              "and independent 15- or 300-epoch runs. Hyperparameters are selected without test-set feedback. ")
    if status["report_complete"]:
        lr_winner=min((results[n] for n in EXPECTED[:3]),key=lambda r:(r["final"]["train_loss"],r["config"]["experiment"]["lr"]))
        delta=100*(results["schedule_cosine"]["final"]["val_accuracy"]-results["schedule_constant"]["final"]["val_accuracy"])
        abstract+=(f"The initial learning rate minimizing final training cross-entropy is {lr_winner['config']['experiment']['lr']:g}. "
                   f"Cosine annealing changes final validation accuracy by {delta:+.2f} percentage points relative to a constant rate. "
                   f"The validation-selected model, \\texttt{{{tex(test['name'])}}}, achieves {100*test['metrics']['accuracy']:.2f}\\% "
                   f"hold-out top-1 accuracy on {test['metrics']['n']} images.")
        if has_overlap:
            abstract+=" Byte-identical overlap in the supplied splits also limits the independence of the held-out estimate."
        if has_conflicts:
            abstract+=" Conflicting annotations on identical images are retained and explicitly audited as label noise."
    else:
        abstract+=("This is an incomplete experimental draft: the required full-data results and final "
                   "hold-out evaluation are not yet all available. Missing values are not estimates; "
                   "synthetic software-verification results are excluded.")
    (output/"abstract.tex").write_text(abstract+"\n")
    if prepared:
        s = prepared["statistics"]
        stat_text = ("The training-set channel means (R, G, B) are "
                     f"({', '.join(f'{x:.6f}' for x in s['mean'])}); the population standard deviations are "
                     f"({', '.join(f'{x:.6f}' for x in s['std'])}).\n")
        stat_text+=quality_text+"\n" if quality_text else ""
    else:
        stat_text = ("Training-set channel means and standard deviations have not yet been measured: "
                     "the assignment dataset is unavailable in the current environment. "
                     "ImageNet statistics are not substituted.\n")
    (output / "statistics.tex").write_text(stat_text)
    if prepared and "counts" in prepared:
        counts = prepared["counts"]
        original = prepared.get("source_counts", counts)
        data_summary = (
            f"\\newcommand{{\\TrainingCount}}{{{counts['train']}}}\n"
            f"The supplied dataset contains {sum(original.values()):,} images across 120 dog breeds: "
            f"{original['train']:,} training, {original['val']:,} validation, and {original['test']:,} test images. "
        )
        if prepared.get("cleaning"):
            data_summary += (f"After removing duplicates and conflicting-label records, the retained splits contain "
                             f"{counts['train']:,}, {counts['val']:,}, and {counts['test']:,} images, respectively. "
                             "Normalization statistics and all reported metrics use these retained splits.\n")
    else:
        data_summary = (r"\newcommand{\TrainingCount}{n_{\mathrm{train}}}"+"\n"
                        "The dataset contains 120 dog breeds. Retained split sizes will be reported after preprocessing.\n")
    (output / "data_summary.tex").write_text(data_summary)
    table = [r"\begin{table*}[t]", r"\centering", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Experiment & Epochs & Train CE & Val. CE & Train acc. (\%) & Val. acc. (\%) \\", r"\midrule"]
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["experiment", "epochs", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
    for name in EXPECTED:
        r = results.get(name)
        if r:
            f = r["final"]
            values = [f["epoch"], loss_number(f['train_loss']), loss_number(f['val_loss']),
                      f"{100*f['train_accuracy']:.2f}", f"{100*f['val_accuracy']:.2f}"]
            writer.writerow([name, f["epoch"],f["train_loss"],f["val_loss"],f["train_accuracy"],f["val_accuracy"]])
        else:
            values = ["--"]*5
            writer.writerow([name]+[""]*5)
        table.append(tex(name) + " & " + " & ".join(map(str,values)) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Final-epoch clean-image metrics. Training and validation use the same unaugmented evaluation protocol."+
              ("" if all_complete else " A dash denotes an unexecuted experiment, not zero.")+"}",
              r"\label{tab:results}", r"\end{table*}"]
    (output / "results_table.tex").write_text("\n".join(table)+"\n")
    (output / "results.csv").write_text(stream.getvalue())
    groups = {"lr": EXPECTED[:3], "schedule": EXPECTED[3:5], "decay": [EXPECTED[4]]+EXPECTED[5:7]}
    if (root / "mixup_baseline.json").exists():
        groups["mixup"] = [load_json(root/"mixup_baseline.json")["name"],"mixup"]
    else:
        groups["mixup"] = ["unselected_baseline","mixup"]
    for name, names in groups.items():
        text = observations(results, names)+measured_interpretation(name,results,names,root)
        if name == "lr" and len(results.keys() & set(names)) == 3:
            winner = min((results[n] for n in names),key=lambda r:(r["final"]["train_loss"],r["config"]["experiment"]["lr"]))
            text += f"The initial learning rate used in subsequent stages is {winner['config']['experiment']['lr']:g}, selected by minimum final training CE as required.\n"
        if name == "mixup" and "mixup" in results:
            e = results["mixup"]["config"]["experiment"]
            text += (f"The selected pre-mixup baseline uses initial learning rate {e['lr']:g}, "
                     f"the {e['schedule']} schedule, and weight decay {e['weight_decay']:g}.\n")
        (output / f"{name}_observations.tex").write_text(text)
    if test:
        m = test["metrics"]
        text = (f"The model selected using validation results is \\texttt{{{tex(test['name'])}}}. "
                f"Its final hold-out top-1 accuracy is {100*m['accuracy']:.2f}\\% "
                f"({m['n']} images), with cross-entropy {m['loss']:.4f}. "
                "The test result was obtained only after freezing the validation-based choice.\n")
        if has_overlap:
            text+="This assignment-split score must be interpreted with the byte-identical overlap reported in preprocessing; it is not an independent unseen-image estimate.\n"
        if has_conflicts:
            text+="The supplied annotation conflicts documented in preprocessing can also affect model fitting and score interpretation.\n"
    else:
        text = "The hold-out test set has not been evaluated. No test accuracy is claimed.\n"
    (output / "test.tex").write_text(text)
    if status["report_complete"]:
        runtime=results["mixup"].get("runtime",{})
        precision=results["mixup"]["config"].get("precision","fp32")
        limitations=(f"The recorded runtime uses PyTorch {tex(runtime.get('torch','unknown'))} and "
                     f"torchvision {tex(runtime.get('torchvision','unknown'))} on {tex(runtime.get('gpu') or runtime.get('device','unknown'))}. "
                     f"Training precision is {tex(precision)}; clean training, validation, and test evaluation use FP32. "
                     "The observed rankings come from one initialization seed and one data split. "
                     "They do not quantify run-to-run variance or establish statistical significance. "
                     "Class imbalance and domain shift can affect the aggregate hold-out accuracy. "
                     "Checkpoint, optimizer, scheduler, and random-number states are retained for reproducibility. "
                     "Cross-hardware bitwise identity is not assumed.")
    else:
        limitations=("The required full-data experiment matrix and final hold-out evaluation are not yet complete. "
                     "The report cannot establish dog-breed classification performance until those runs finish. "
                     "No synthetic metrics are substituted for missing observations. The accompanying smoke "
                     "workflow validates software only. Resume the one-command workflow to populate the measured results.")
    (output/"limitations.tex").write_text(limitations+" "+quality_text+"\n")
    print(notice)
