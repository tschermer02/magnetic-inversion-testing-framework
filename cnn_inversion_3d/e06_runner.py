"""Training services for E06, built on the E05 workflow."""
from __future__ import annotations
import hashlib,json,platform
from datetime import datetime,timezone
from pathlib import Path
import tensorflow as tf
from cnn_inversion_3d.dataset import TMI_SHAPE,build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e04_forward import E04TMIForward
from cnn_inversion_3d.e05_callbacks import GradientDiagnostics,ResumeCheckpoint
from cnn_inversion_3d.e05_runner import _dataset_hashes,_git_state
from cnn_inversion_3d.e06_callbacks import DualMetricCheckpoint
from cnn_inversion_3d.e06_config import E06SuiteConfig,E06Variant
from cnn_inversion_3d.e01_training import E01LossConfig,E01TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model

def loss_config(variant):
    return E01LossConfig(lambda_susceptibility=100,body_fraction=variant.body_fraction,
        lambda_depth=2,alpha_center=1,lambda_sensitivity=1,lambda_amplitude=1,
        lambda_body_susceptibility=0,lambda_tmi=1e-4,lambda_tversky=.1,
        tversky_alpha=.7,tversky_beta=.3,occupancy_threshold=.001,occupancy_sharpness=1000,
        occupancy_mode=variant.occupancy_mode,occupancy_tau_si=.001)

def build_model(config,variant):
    _,weights=build_e01_sensitivity_weights()
    return E01TrainingModel(build_e01_model(ModelConfig(base_filters=config.base_filters)),weights,E04TMIForward(),
        tmi_scale=config.tmi_scale_nt,loss_config=loss_config(variant))

def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def _write(path,value):path.write_text(json.dumps(value,indent=2,default=str),encoding="utf-8")

def train_variant(config:E06SuiteConfig,variant:E06Variant,root:Path,*,resume=False):
    output=root/variant.identifier;output.mkdir(parents=True,exist_ok=True)
    run={"suite":config.to_dict(),"variant":variant.to_dict(),"loss":vars(loss_config(variant)),
        "dataset_hashes":_dataset_hashes(config.dataset),"selection":"fixed 50/50 balanced validation MAE", "git":_git_state()}
    compatibility=_hash(run);path=output/"run_config.json";status=output/"stage_status.json"
    if path.exists() and json.loads(path.read_text())["compatibility_hash"]!=compatibility:raise ValueError(f"Incompatible existing E06 output: {output}")
    if not path.exists():run.update({"compatibility_hash":compatibility,"python":platform.python_version(),"tensorflow":tf.__version__});_write(path,run)
    if resume and status.exists() and json.loads(status.read_text()).get("stage")=="complete":return output
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(config.seed)
    train,validation,_,counts=build_training_datasets(dataset_directory=config.dataset,batch_size=config.batch_size,
        tmi_scale=config.tmi_scale_nt,susceptibility_scale=config.susceptibility_scale_si,random_seed=config.seed)
    model=build_model(config,variant);model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate),jit_compile=False)
    model(tf.zeros((1,*TMI_SHAPE),tf.float32),training=False);model.optimizer.build(model.inversion_model.trainable_variables)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=model.optimizer);manager=tf.train.CheckpointManager(checkpoint,str(output/"resume_checkpoint"),max_to_keep=1)
    initial=0;selection_state=None
    if resume and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        if status.exists():initial=int(json.loads(status.read_text()).get("last_completed_epoch",0))
        selection_path=output/"checkpoint_selection.json"
        if selection_path.exists():selection_state=json.loads(selection_path.read_text())
    fixed=next(iter(train.take(1)));selector=DualMetricCheckpoint(validation,output,config.support_threshold_si,config.early_stopping_patience,selection_state)
    callbacks=[selector,ResumeCheckpoint(manager,status),GradientDiagnostics((fixed[0][:config.diagnostic_samples],fixed[1][:config.diagnostic_samples]),output/"gradient_diagnostics.jsonl",config.diagnostic_interval),
        tf.keras.callbacks.CSVLogger(output/"training_history.csv",append=initial>0)]
    _write(status,{"stage":"training","last_completed_epoch":initial,"counts":counts,"resume_mode":"full model and optimizer" if initial else "fresh"})
    model.fit(train,validation_data=validation,epochs=config.epochs,initial_epoch=initial,callbacks=callbacks,shuffle=False)
    state=json.loads((output/"checkpoint_selection.json").read_text());_write(status,{"stage":"complete","primary_epoch":state["primary_epoch"],
        "geometry_epoch":state["geometry_epoch"],"primary_score":state["best_primary"],"geometry_iou":state["best_geometry"]})
    return output
