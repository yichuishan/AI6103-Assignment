"""One-command AutoDL workflow. Installs prerequisites, prepares data, trains and exports."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run(argv, cwd=REPO, env=None):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def status(work, state, **details):
    write_json(work/"status.json", {"state":state,"pid":os.getpid(),**details})


def ensure_dependencies(skip_install, smoke):
    if not skip_install:
        probe = subprocess.run([sys.executable,"-c",
                                "import torch,torchvision; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cuda_ok = probe.returncode == 0
        if not smoke and not cuda_ok:
            run([sys.executable,"-m","pip","install","--disable-pip-version-check",
                 "torch==2.6.0","torchvision==0.21.0","--index-url",
                 "https://download.pytorch.org/whl/cu124"])
        run([sys.executable,"-m","pip","install","--disable-pip-version-check","-r",REPO/"requirements.txt"])
    # Validate imports in a new process, avoiding stale modules after installation.
    run([sys.executable,"-c",
         "import torch,torchvision,numpy,PIL,matplotlib,scipy; "
         "print('torch:',torch.__version__,'torchvision:',torchvision.__version__); "
         "print('CUDA:',torch.cuda.is_available())"])


def ensure_tex(skip_install):
    if all(shutil.which(n) for n in ("pdflatex","bibtex","pdfinfo","kpsewhich")):
        packages=("times.sty","ptmr7t.tfm","natbib.sty","caption.sty","booktabs.sty","listings.sty","placeins.sty")
        if all(subprocess.run(["kpsewhich",name],stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode==0 for name in packages):
            return
    if skip_install or os.geteuid() != 0 or not shutil.which("apt-get"):
        raise RuntimeError("PDF tools missing. Install texlive-latex-extra texlive-fonts-recommended poppler-utils, then rerun.")
    env = dict(os.environ,DEBIAN_FRONTEND="noninteractive")
    run(["apt-get","update"],env=env)
    run(["apt-get","install","-y","--no-install-recommends","texlive-latex-base",
         "texlive-latex-recommended","texlive-latex-extra","texlive-fonts-recommended","poppler-utils"],env=env)


def choose_precision(requested, smoke):
    if smoke:
        return "fp32"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available in this Python environment")
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU; choose --precision fp16 or fp32")
    return requested


def check_gpu_batch(precision):
    """Check the actual required 128-image batch before paying for a long run."""
    import torch
    from torchvision.models import efficientnet_v2_s
    print("GPU:",torch.cuda.get_device_name(0),"precision:",precision,flush=True)
    model = efficientnet_v2_s(weights=None,num_classes=120).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
    x = torch.randn(128,3,100,100,device="cuda")
    y = torch.arange(128,device="cuda") % 120
    with torch.autocast("cuda",dtype=torch.bfloat16 if precision=="bf16" else torch.float16,
                        enabled=precision!="fp32"):
        loss = torch.nn.functional.cross_entropy(model(x),y)
    if precision=="fp16":
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print(f"Batch 128 forward/backward/SGD passed; peak allocated GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB",flush=True)
    del optimizer,model,x,y,loss
    torch.cuda.empty_cache()


def copy_report(work):
    report = work/"report"
    report.mkdir(parents=True,exist_ok=True)
    for name in ("report.tex","references.bib","aaai24.sty","aaai24.bst","STYLE_SOURCE.md"):
        destination = report/name
        # Preserve an edited report on subsequent runs.
        if not destination.exists():
            shutil.copy2(REPO/"report"/name,destination)
    return report


def check_input_binding(work,binding):
    """Permit correcting a failed preparation, never reuse another trained run."""
    saved=work/"input.json"
    if not saved.exists() or json.loads(saved.read_text())==binding:
        return
    has_runs=any(p.is_file() for p in (work/"runs").rglob("*"))
    if (work/"prepared.json").exists() or has_runs:
        raise ValueError("Output directory belongs to another data/configuration. Choose a new --output directory.")
    previous=saved.read_bytes()
    backup=work/("input-before-failed-preparation-"+hashlib.sha256(previous).hexdigest()[:12]+".json")
    if not backup.exists():
        backup.write_bytes(previous)
    print("Updating input configuration after unsuccessful data preparation; previous configuration preserved:",backup,flush=True)


def preflight(args,work):
    status(work,"preflight")
    ensure_dependencies(args.skip_install,args.smoke)
    ensure_tex(args.skip_install)
    from ai6103.autodata import extract_zip, find_layout, locate_dataset, file_sha256
    from ai6103.data import CLEANING_POLICY, export_cleaning, load_json, prepare
    precision = choose_precision(args.precision,args.smoke)
    if args.smoke:
        dataset = work/"synthetic"
        if not dataset.exists():
            run([sys.executable,REPO/"scripts/make_smoke_data.py","--output",dataset])
        root, manifest, resplit = dataset,None,False
        source = {"path":str(dataset),"smoke":True}
    else:
        dataset = locate_dataset(args.dataset)
        if dataset.is_file():
            if not zipfile.is_zipfile(dataset):
                raise ValueError("Dataset file must be ZIP; alternatively pass an extracted directory")
            print("Validating and extracting dataset ZIP...",flush=True)
            base,digest = extract_zip(dataset,work)
            source = {"path":str(dataset),"sha256":digest,"smoke":False}
        else:
            base=dataset
            source = {"path":str(dataset),"smoke":False}
        root,manifest,resplit = find_layout(base,work,args.manifest)
    binding={"source":source,"root":str(root),"manifest":str(manifest) if manifest else None,
             "manifest_sha256":file_sha256(manifest) if manifest else None,
             "resplit":resplit,"seed":args.seed,"precision":precision,"workers":args.workers,
             "cleaning_policy":CLEANING_POLICY}
    saved=work/"input.json"
    check_input_binding(work,binding)
    prepared=work/"prepared.json"
    if not prepared.exists():
        print("Preparing split validation and training-only RGB statistics...",flush=True)
        prepare(root,prepared,manifest,resplit,args.seed,strict=not args.smoke)
    p=load_json(prepared)
    if p.get("cleaning",{}).get("policy") != CLEANING_POLICY:
        raise ValueError("Prepared data use the old preprocessing policy. Choose a new --output directory (results_cleaned).")
    source_images = p["rows"] + p["cleaning"]["removed_images"]
    if any(file_sha256(r["path"]) != r["sha256"] for r in source_images):
        raise ValueError("Prepared dataset images have changed; use a new output directory")
    export_cleaning(p, work)
    write_json(saved,binding)
    if "data_quality_audit" in p:
        write_json(work/"data_quality_audit.json",p["data_quality_audit"])
        print("DATA_QUALITY_AUDIT:",work/"data_quality_audit.json",
              p["data_quality_audit"]["overlap_images"],flush=True)
    print("SPLIT_COUNTS:",p["counts"],"RGB_STATISTICS:",p["statistics"],flush=True)
    print("CLEANING_SUMMARY:",work/"cleaning_summary.json",
          "REMOVED_COUNTS:",p["cleaning"]["removed_counts"],flush=True)
    if not args.smoke:
        check_gpu_batch(precision)
    copy_report(work)
    # Build draft early to validate style, fonts and TeX before training.
    # Keep this separate so a check-only invocation never replaces a final report.
    preflight_work=work/".preflight"
    preflight_report=copy_report(preflight_work)
    from ai6103.reporting import generate
    generate(preflight_work/"missing.json",preflight_work/"empty-runs",preflight_report/"generated")
    run([sys.executable,"-m","ai6103.cli","theory","--output",preflight_work/"artifacts/figures"])
    run(["bash",REPO/"scripts/build_report.sh",preflight_report])
    run([sys.executable,"-m","pytest","-q"],env=dict(os.environ,MPLCONFIGDIR=str(work/".mpl")))
    return {"work":str(work),"precision":precision,"smoke":args.smoke,
            "workers":args.workers,"threads":args.threads,"seed":args.seed,
            "author":args.author,"repo":str(REPO)}


def execute(config):
    work=Path(config["work"])
    status(work,"running",smoke=config["smoke"])
    from ai6103.data import load_json
    from ai6103.engine import run_pipeline, test_once
    from ai6103.reporting import generate
    from ai6103.visualize import curves, theoretical_figures
    import torch
    torch.set_num_threads(config["threads"])
    prepared=load_json(work/"prepared.json")
    device="cpu" if config["smoke"] else "cuda"
    runs=work/"runs"
    run_pipeline(prepared,runs,device=device,workers=config["workers"],seed=config["seed"],
                 smoke=config["smoke"],precision=config["precision"])
    result=test_once(prepared,runs,device=device,workers=config["workers"])
    print("Final evaluation:",json.dumps(result),flush=True)
    curves(runs,work/("artifacts/verification/smoke_figures" if config["smoke"] else "artifacts/figures"))
    if config["smoke"]:
        # Keep the formal report empty and explicitly incomplete for this fixture.
        theoretical_figures(work/"artifacts/figures")
        generate(work/"missing.json",work/"empty-runs",work/"report/generated")
    else:
        generate(work/"prepared.json",runs,work/"report/generated",author=config["author"])
    run(["bash",REPO/"scripts/build_report.sh",work/"report"])
    package=work/("AI6103_SMOKE_VERIFICATION.zip" if config["smoke"] else "AI6103_results.zip")
    with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
        for folder in (work/"report",work/"artifacts",runs):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix in {".tex",".bib",".bst",".sty",".md",".pdf",".png",".csv",".json"}:
                    archive.write(path,path.relative_to(work))
        archive.write(work/"input.json","input.json")
        if (work/"data_quality_audit.json").exists():
            archive.write(work/"data_quality_audit.json","data_quality_audit.json")
        for name in ("cleaning_summary.json","cleaned_manifest.csv","removed_images.csv"):
            archive.write(work/name,name)
    status(work,"complete",smoke=config["smoke"],package=str(package),
           report=str(work/"report/report.pdf"))
    print("COMPLETE:",package,"\nREPORT:",work/"report/report.pdf",flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",nargs="?",help="ZIP or extracted directory; default auto-detect /root/autodl-tmp/data.zip")
    parser.add_argument("--output",type=Path,default=REPO/"results_cleaned")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--precision",choices=["auto","fp32","bf16","fp16"],default="auto")
    parser.add_argument("--author",default="AI6103 Assignment",help="Report author name")
    parser.add_argument("--foreground",action="store_true",help="Keep training attached to this terminal")
    parser.add_argument("--check-only",action="store_true",help="Install/check/prepare without running full experiments")
    parser.add_argument("--skip-install",action="store_true",help="Require existing Python and PDF dependencies")
    parser.add_argument("--smoke",action="store_true",help="Synthetic CPU verification only, not assignment results")
    parser.add_argument("--worker-config",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd",type=int,help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.worker_config:
        config=json.loads(args.worker_config.read_text())
        work=Path(config["work"])
        fd=args.lock_fd
        if fd is None or os.fstat(fd).st_ino != (work/".pipeline.lock").stat().st_ino:
            raise RuntimeError("Missing inherited workflow lock")
    else:
        work=args.output.expanduser().resolve()
        work.mkdir(parents=True,exist_ok=True)
        lock=(work/".pipeline.lock").open("a+")
        fd=lock.fileno()
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A workflow is already running in {work}. See pipeline.log and status.json")
    os.environ.setdefault("MPLCONFIGDIR",str(work/".mpl"))
    os.environ.setdefault("XDG_CACHE_HOME",str(work/".cache"))
    try:
        if not args.worker_config:
            config=preflight(args,work)
            config_path=work/"worker_config.json"
            write_json(config_path,config)
            if args.check_only:
                status(work,"preflight_complete")
                print("All preflight checks passed; no full experiment was started.",flush=True)
                return
            if not args.foreground:
                with (work/"pipeline.log").open("a",buffering=1) as log:
                    child=subprocess.Popen([sys.executable,"-u",__file__,"--worker-config",str(config_path),
                                            "--lock-fd",str(fd)],cwd=REPO,stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
                print(f"Background workflow launched (PID {child.pid}). You may close the terminal.\n"
                      f"Monitor: tail -f {work/'pipeline.log'}\n"
                      f"Final package: {work/('AI6103_SMOKE_VERIFICATION.zip' if args.smoke else 'AI6103_results.zip')}\n"
                      "Rerun the same command after interruption to resume.",flush=True)
                return
        execute(config)
    except BaseException as error:
        status(work,"failed",error=str(error))
        raise
    finally:
        if args.worker_config:
            os.close(fd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. The next invocation resumes complete epochs.",file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        traceback.print_exc()
        print("FAILED:",error,file=sys.stderr)
        sys.exit(1)
