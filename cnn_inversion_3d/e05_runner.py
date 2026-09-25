"""Training and parent-selection services for the E05 suite."""
from __future__ import annotations
import csv,hashlib,json,platform,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.dataset import TMI_SHAPE,build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e04_forward import E04TMIForward
from cnn_inversion_3d.e05_callbacks import GradientDiagnostics,PrimaryMetricCheckpoint,ResumeCheckpoint
from cnn_inversion_3d.e05_config import E05SuiteConfig,E05Variant,FIRST_STAGE,dependent_variants
from cnn_inversion_3d.e05_training import E05LossConfig,E05TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import DisabledTMIForward,_git_state,_sha256

def _utc():return datetime.now(timezone.utc).isoformat()
def _json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def _write(path,value):path.write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")

def loss_config(variant:E05Variant):
    return E05LossConfig(lambda_susceptibility=variant.susceptibility_multiplier,
        lambda_depth=2,alpha_center=1,lambda_sensitivity=1,lambda_amplitude=1,
        lambda_body_susceptibility=0,lambda_tmi=variant.tmi_coefficient,
        lambda_tversky=.1,tversky_alpha=.7,tversky_beta=.3,
        occupancy_threshold=.001,occupancy_sharpness=1000,occupancy_mode="legacy_threshold_sigmoid",
        lambda_vertical_gradient=variant.vertical_gradient_coefficient,susceptibility_reference_si=.1)

def build_model(config:E05SuiteConfig,variant:E05Variant):
    _,weights=build_e01_sensitivity_weights()
    forward=E04TMIForward() if variant.tmi_coefficient else DisabledTMIForward()
    return E05TrainingModel(build_e01_model(ModelConfig(base_filters=config.base_filters)),weights,
        forward,tmi_scale=config.tmi_scale_nt,loss_config=loss_config(variant))

def _dataset_hashes(dataset):
    result={}
    for name in ("metadata.json","train_manifest.csv","validation_manifest.csv","test_manifest.csv"):
        path=dataset/name
        if path.is_file():result[name]=_sha256(path)
    return result

def train_variant(config:E05SuiteConfig,variant:E05Variant,root:Path,*,resume=False):
    output=root/variant.identifier;output.mkdir(parents=True,exist_ok=True)
    hashes=_dataset_hashes(config.dataset)
    run={"suite":config.to_dict(),"variant":variant.to_dict(),"dataset_hashes":hashes,
        "selection_rule":"minimum per-sample validation true-body MAE; tie-break higher support IoU",
        "git":_git_state()}
    compatibility=_json_hash(run);config_path=output/"run_config.json";status_path=output/"stage_status.json"
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("compatibility_hash")!=compatibility:
            raise ValueError(f"Incompatible existing E05 output: {output}")
    else:
        run.update({"compatibility_hash":compatibility,"environment":{"python":platform.python_version(),
            "tensorflow":tf.__version__,"gpus":[device.name for device in tf.config.list_physical_devices("GPU")]},
            "created_utc":_utc()});_write(config_path,run)
    if resume and status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("stage")=="complete":
        print(f"Skipping completed compatible variant {variant.identifier}");return output
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(config.seed)
    train,validation,_,counts=build_training_datasets(dataset_directory=config.dataset,
        batch_size=config.batch_size,tmi_scale=config.tmi_scale_nt,
        susceptibility_scale=config.susceptibility_scale_si,random_seed=config.seed)
    model=build_model(config,variant);optimizer=tf.keras.optimizers.Adam(config.learning_rate)
    model.compile(optimizer=optimizer,jit_compile=False);model(tf.zeros((1,*TMI_SHAPE),tf.float32),training=False)
    optimizer.build(model.inversion_model.trainable_variables)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    manager=tf.train.CheckpointManager(checkpoint,str(output/"resume_checkpoint"),max_to_keep=1)
    initial_epoch=0;resume_note="fresh initialization"
    primary_state=None
    if resume and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if status_path.exists():initial_epoch=int(json.loads(status_path.read_text(encoding="utf-8")).get("last_completed_epoch",0))
        selection_path=output/"primary_checkpoint.json"
        if selection_path.exists():primary_state=json.loads(selection_path.read_text(encoding="utf-8"))
        resume_note="model and Adam optimizer restored; tf.data reshuffle sequence restarts at process boundary"
    fixed_batch=next(iter(train.take(1)))
    primary=PrimaryMetricCheckpoint(validation,output,config.support_threshold_si,
        config.early_stopping_patience,primary_state)
    composite=tf.keras.callbacks.ModelCheckpoint(output/"best_composite_loss.weights.h5",monitor="val_loss",
        mode="min",save_best_only=True,save_weights_only=True,verbose=1)
    history_path=output/"training_history.csv"
    if initial_epoch and history_path.is_file():
        with history_path.open(newline="",encoding="utf-8") as stream:
            prior=list(csv.DictReader(stream))
        finite=[float(row["val_loss"]) for row in prior if row.get("val_loss")]
        if finite:composite.best=min(finite)
    callbacks=[primary,composite,
        ResumeCheckpoint(manager,status_path),
        GradientDiagnostics((fixed_batch[0][:config.diagnostic_samples],fixed_batch[1][:config.diagnostic_samples]),
            output/"gradient_diagnostics.jsonl",config.diagnostic_interval),
        tf.keras.callbacks.CSVLogger(history_path,append=initial_epoch>0)]
    _write(status_path,{"stage":"training","started_utc":_utc(),"initial_epoch":initial_epoch,
        "resume_note":resume_note,"counts":counts})
    model.fit(train,validation_data=validation,epochs=config.epochs,initial_epoch=initial_epoch,
        callbacks=callbacks,shuffle=False)
    selected=output/"selected.weights.h5"
    if not selected.is_file():raise FileNotFoundError(f"Primary checkpoint missing: {selected}")
    model.load_weights(selected);model.inversion_model.save(output/"selected_model.keras")
    state=json.loads((output/"primary_checkpoint.json").read_text(encoding="utf-8"))
    _write(status_path,{"stage":"complete","completed_utc":_utc(),"best_epoch":state["best_epoch"],
        "selected_checkpoint":str(selected.resolve()),"resume_note":resume_note})
    return output

def select_parent(root:Path,*,resume=False):
    manifest_path=root/"parent_selection.json";candidates=[]
    for identifier,variant in FIRST_STAGE.items():
        path=root/identifier/"primary_checkpoint.json"
        if not path.is_file():raise FileNotFoundError(f"Cannot select E05 parent; missing {path}")
        state=json.loads(path.read_text(encoding="utf-8"));candidates.append({"identifier":identifier,
            "susceptibility_multiplier":variant.susceptibility_multiplier,
            "validation_true_body_mae_si":state["best_mae"],"validation_support_iou":state["best_iou"],
            "selected_epoch":state["best_epoch"]})
    for rank,item in enumerate(sorted(candidates,key=lambda x:x["validation_true_body_mae_si"]),1):item["mae_rank"]=rank
    for rank,item in enumerate(sorted(candidates,key=lambda x:-x["validation_support_iou"]),1):item["iou_rank"]=rank
    for item in candidates:item["rank_sum"]=item["mae_rank"]+item["iou_rank"]
    winner=min(candidates,key=lambda x:(x["rank_sum"],x["validation_true_body_mae_si"],x["susceptibility_multiplier"]))
    decision={"frozen":True,"selection_split":"validation","rule":"lowest MAE-rank + IoU-rank; ties lower MAE then multiplier",
        "candidates":candidates,"selected_parent":winner,"timestamp_utc":_utc()}
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["selected_parent"]["identifier"]!=winner["identifier"]:
            raise ValueError("Frozen E05 parent conflicts with current first-stage results")
        return FIRST_STAGE[existing["selected_parent"]["identifier"]]
    _write(manifest_path,decision);return FIRST_STAGE[winner["identifier"]]
