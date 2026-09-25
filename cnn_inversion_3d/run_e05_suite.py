"""Unattended training-to-report workflow for the controlled E05 suite."""
from __future__ import annotations
import argparse,json,traceback
from datetime import datetime,timezone
from pathlib import Path
from cnn_inversion_3d.e05_config import E05SuiteConfig,FIRST_STAGE,dependent_variants,with_seed
from cnn_inversion_3d.e05_runner import select_parent,train_variant
from evaluation.e05_evaluate import evaluate_variant
from evaluation.e05_report import build_suite_report

ALL_IDS=("e05a","e05b","e05c","e05d","e05e","e05f")
def _parse(value):
    if value=="all":return list(ALL_IDS)
    result=[item.strip().lower() for item in value.split(",") if item.strip()]
    unknown=set(result)-set(ALL_IDS)
    if unknown:raise ValueError(f"Unknown E05 variants: {sorted(unknown)}")
    return result

def run_e05_suite(*,variants="all",seed=20260727,resume=False,plots="all",stage="all",
                  split="test",continue_on_error=False,dataset=None,evaluation_manifest=None):
    config=with_seed(E05SuiteConfig(),seed)
    if dataset is not None:config=__import__('dataclasses').replace(config,dataset=Path(dataset))
    requested=_parse(variants);output_root=Path("outputs/E05")/f"seed_{seed}"
    evaluation_root=Path("prediction_outputs/E05")/f"seed_{seed}"
    report_root=Path("analysis_outputs/E05")/f"seed_{seed}";output_root.mkdir(parents=True,exist_ok=True)
    requested_manifest=Path(evaluation_manifest) if evaluation_manifest else Path(f"{split}_manifest.csv")
    manifest_path=requested_manifest if requested_manifest.is_absolute() else config.dataset/requested_manifest
    evaluation_tag=requested_manifest.stem if evaluation_manifest else split
    required=(config.dataset/"metadata.json",config.dataset/"train_manifest.csv",
        config.dataset/"validation_manifest.csv",manifest_path)
    missing=[str(path) for path in required if not path.is_file()]
    if missing:raise FileNotFoundError(f"E05 required dataset files missing: {missing}")
    failures=[];trained={}
    def write_failure_status():
        (output_root/"suite_status.json").write_text(json.dumps({"requested_variants":requested,
            "seed":seed,"stage":stage,"split":split,"failures":failures,
            "failed_utc":datetime.now(timezone.utc).isoformat()},indent=2),encoding="utf-8")
    def attempt(identifier,variant):
        try:trained[identifier]=train_variant(config,variant,output_root,resume=resume)
        except Exception as error:
            failures.append({"variant":identifier,"stage":"training","error":repr(error),"traceback":traceback.format_exc()})
            write_failure_status()
            if not continue_on_error:raise
    needs_dependent=any(item in requested for item in ("e05d","e05e","e05f"))
    training_requested=stage in ("all","train")
    if training_requested:
        first_to_run=list(FIRST_STAGE) if variants=="all" or needs_dependent else [item for item in requested if item in FIRST_STAGE]
        for identifier in first_to_run:attempt(identifier,FIRST_STAGE[identifier])
    if needs_dependent or variants=="all":
        try:parent=select_parent(output_root,resume=resume)
        except Exception as error:
            failures.append({"variant":None,"stage":"parent_selection","error":repr(error),"traceback":traceback.format_exc()})
            write_failure_status();raise
    else:
        manifest=output_root/"parent_selection.json"
        parent=FIRST_STAGE[json.loads(manifest.read_text(encoding="utf-8"))["selected_parent"]["identifier"]] if manifest.is_file() else FIRST_STAGE["e05a"]
    resolved={**FIRST_STAGE,**dependent_variants(parent)}
    if training_requested:
        for identifier in [item for item in requested if item in ("e05d","e05e","e05f")]:attempt(identifier,resolved[identifier])
    if stage in ("all","predict","analyze"):
        for identifier in requested:
            try:
                training_output=output_root/identifier
                evaluation_output=evaluation_root/identifier/evaluation_tag
                evaluate_variant(config,resolved[identifier],training_output,evaluation_output,
                    split=evaluation_tag,plots=plots,manifest_name=str(requested_manifest))
            except Exception as error:
                failures.append({"variant":identifier,"stage":"evaluation","error":repr(error),"traceback":traceback.format_exc()})
                write_failure_status()
                if not continue_on_error:raise
        if all((evaluation_root/item/evaluation_tag/"per_sample_metrics.csv").is_file() for item in ALL_IDS):
            build_suite_report(evaluation_root,resolved,parent.identifier,report_root/evaluation_tag,split=evaluation_tag)
    status={"requested_variants":requested,"seed":seed,"stage":stage,"split":split,"resume":resume,
        "evaluation_manifest":str(requested_manifest),"parent":parent.to_dict(),"failures":failures,
        "completed_utc":datetime.now(timezone.utc).isoformat()}
    (output_root/"suite_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    if failures:raise RuntimeError(f"E05 suite completed with {len(failures)} failure(s); see {output_root/'suite_status.json'}")
    return status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",default="all",help="all or comma-separated e05a...e05f")
    parser.add_argument("--seed",type=int,default=20260727);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--plots",choices=("all","none"),default="all")
    parser.add_argument("--stage",choices=("all","train","predict","analyze"),default="all")
    parser.add_argument("--split",choices=("validation","test"),default="test")
    parser.add_argument("--evaluation-manifest",type=Path,
        help="Optional fresh manifest; outputs use its stem and never replace existing test results.")
    parser.add_argument("--dataset",type=Path);parser.add_argument("--continue-on-error",action="store_true")
    args=parser.parse_args();run_e05_suite(**vars(args))

if __name__=="__main__":main()
