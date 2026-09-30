"""One-command sequential E06 training, dual evaluation, and reporting."""
from __future__ import annotations
import argparse,json,traceback
from datetime import datetime,timezone
from pathlib import Path
from cnn_inversion_3d.e06_config import E06SuiteConfig,VARIANTS,with_seed
from cnn_inversion_3d.e06_runner import build_model,train_variant
from evaluation.e05_evaluate import evaluate_variant
from evaluation.e06_report import build_report

def _parse(value):
    ids=list(VARIANTS) if value=="all" else [x.strip().lower() for x in value.split(",") if x.strip()]
    unknown=set(ids)-set(VARIANTS)
    if unknown:raise ValueError(f"Unknown E06 variants: {sorted(unknown)}")
    return ids

def run_e06_suite(*,variants="all",seed=42,resume=False,plots="all",stage="all",dataset=None,limit=None):
    config=with_seed(E06SuiteConfig(),seed)
    if dataset:config=__import__('dataclasses').replace(config,dataset=Path(dataset))
    ids=_parse(variants);training_root=Path("outputs/E06")/f"seed_{seed}";evaluation_root=Path("prediction_outputs/E06")/f"seed_{seed}";report_root=Path("analysis_outputs/E06")/f"seed_{seed}"
    required=[config.dataset/x for x in ("metadata.json","train_manifest.csv","validation_manifest.csv","test_manifest.csv")];missing=[str(x) for x in required if not x.is_file()]
    if missing:raise FileNotFoundError(f"E06 required dataset files missing: {missing}")
    failures=[]
    for identifier in ids:
        variant=VARIANTS[identifier]
        try:
            if stage in ("all","train"):train_variant(config,variant,training_root,resume=resume)
            if stage in ("all","predict","analyze"):
                for checkpoint,filename in (("primary","primary.weights.h5"),("geometry","geometry.weights.h5")):
                    evaluate_variant(config,variant,training_root/identifier,evaluation_root/identifier/checkpoint,
                        plots=plots,limit=limit,checkpoint_name=filename,checkpoint_label=checkpoint,model_builder=build_model)
        except Exception as error:
            failures.append({"variant":identifier,"error":repr(error),"traceback":traceback.format_exc()});raise
    if stage in ("all","predict","analyze") and set(ids)==set(VARIANTS):
        build_report(evaluation_root,list(VARIANTS),report_root/"primary","primary")
        build_report(evaluation_root,list(VARIANTS),report_root/"geometry","geometry")
    status={"variants":ids,"seed":seed,"stage":stage,"failures":failures,"completed_utc":datetime.now(timezone.utc).isoformat()};training_root.mkdir(parents=True,exist_ok=True)
    (training_root/"suite_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8");return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--variants",default="all");p.add_argument("--seed",type=int,default=42);p.add_argument("--resume",action="store_true")
    p.add_argument("--plots",choices=("all","none"),default="all");p.add_argument("--stage",choices=("all","train","predict","analyze"),default="all");p.add_argument("--dataset",type=Path);p.add_argument("--limit",type=int)
    run_e06_suite(**vars(p.parse_args()))
if __name__=="__main__":main()
