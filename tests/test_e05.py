import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import pytest
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e05_callbacks import validation_selection_metrics
from cnn_inversion_3d.e05_config import E05SuiteConfig,FIRST_STAGE,dependent_variants
from cnn_inversion_3d.e05_runner import build_model,loss_config,select_parent
from cnn_inversion_3d.e05_training import vertical_gradient_loss_per_sample
from cnn_inversion_3d.model import ModelConfig,build_e01_model

def test_variant_coefficients_and_exact_susceptibility_scaling():
    assert [FIRST_STAGE[key].susceptibility_multiplier for key in ("e05a","e05b","e05c")]==[1,10,100]
    parent=FIRST_STAGE["e05b"];dependent=dependent_variants(parent)
    assert dependent["e05d"].vertical_gradient_coefficient==.1 and dependent["e05d"].tmi_coefficient==0
    assert dependent["e05e"].tmi_coefficient==1e-4 and dependent["e05f"].vertical_gradient_coefficient==.1
    assert loss_config(FIRST_STAGE["e05b"]).lambda_susceptibility==10
    assert loss_config(FIRST_STAGE["e05c"]).lambda_susceptibility==100

def test_vertical_gradient_perfect_zero_and_internal_gap_positive():
    truth=tf.zeros((1,4,2,2,1));truth=tf.tensor_scatter_nd_update(truth,[[0,1,0,0,0],[0,2,0,0,0]],[.05,.05])
    assert float(vertical_gradient_loss_per_sample(truth,truth))==0
    gap=tf.tensor_scatter_nd_update(truth,[[0,2,0,0,0]],[0.])
    assert float(vertical_gradient_loss_per_sample(truth,gap))>0

def test_vertical_gradient_is_finite_and_differentiable():
    truth=tf.ones((1,4,2,2,1))*.02;prediction=tf.Variable(tf.ones_like(truth)*.03)
    with tf.GradientTape() as tape:loss=tf.reduce_mean(vertical_gradient_loss_per_sample(truth,prediction+tf.reshape(tf.range(4,dtype=tf.float32),(1,4,1,1,1))*.001))
    gradient=tape.gradient(loss,prediction)
    assert np.isfinite(float(loss)) and np.all(np.isfinite(gradient.numpy())) and float(tf.norm(gradient))>0

def _selection(path,mae,iou):
    path.mkdir(parents=True);(path/"primary_checkpoint.json").write_text(json.dumps({"best_mae":mae,"best_iou":iou,"best_epoch":1}))

def test_parent_selection_and_frozen_resume(tmp_path):
    _selection(tmp_path/"e05a",.3,.8);_selection(tmp_path/"e05b",.2,.7);_selection(tmp_path/"e05c",.1,.6)
    # All rank sums tie; lower body MAE selects E05c.
    assert select_parent(tmp_path).identifier=="e05c"
    assert select_parent(tmp_path,resume=True).identifier=="e05c"

def test_incompatible_run_config_is_rejected(tmp_path):
    from cnn_inversion_3d.e05_runner import train_variant
    output=tmp_path/"e05a";output.mkdir(parents=True)
    (output/"run_config.json").write_text(json.dumps({"compatibility_hash":"wrong"}))
    with pytest.raises(ValueError,match="Incompatible"):
        train_variant(E05SuiteConfig(dataset=Path("datasets/E01_soft_tversky_full")),FIRST_STAGE["e05a"],tmp_path,resume=True)

def test_short_train_step_prediction_and_selection_metrics_are_finite():
    path=Path("datasets/E01_soft_tversky_full/samples/sample_000000.npz")
    if not path.is_file():pytest.skip("E01 dataset unavailable")
    with np.load(path) as saved:
        tmi=tf.constant(saved["tmi"][None,...,None]/100,tf.float32);truth=tf.constant(saved["susceptibility"][None,...,None],tf.float32)
    config=E05SuiteConfig(base_filters=1);model=build_model(config,FIRST_STAGE["e05a"])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False)
    metrics=model.train_step((tmi,truth));prediction=model.inversion_model(tmi,training=False)
    selected=validation_selection_metrics(model,tf.data.Dataset.from_tensor_slices((tmi,truth)).batch(1))
    assert prediction.shape==(1,24,64,64,1) and all(np.isfinite(float(v)) for v in metrics.values())
    assert np.isfinite(selected["true_body_susceptibility_mae_si"]) and np.isfinite(selected["support_iou"])

def test_one_sample_evaluation_smoke(tmp_path):
    if not Path("datasets/E01_soft_tversky_full/test_manifest.csv").is_file():pytest.skip("E01 dataset unavailable")
    from evaluation.e05_evaluate import evaluate_variant
    config=E05SuiteConfig(base_filters=1);training=tmp_path/"training";training.mkdir()
    model=build_model(config,FIRST_STAGE["e05a"]);model(tf.zeros((1,81,81,1)),training=False)
    model.save_weights(training/"selected.weights.h5")
    rows,summary=evaluate_variant(config,FIRST_STAGE["e05a"],training,tmp_path/"evaluation",
        split="test",plots="all",limit=1)
    assert len(rows)==summary["aggregate"]["sample_count"]==1
    assert np.isfinite(rows[0]["tmi_rmse_nt"])
    assert (tmp_path/"evaluation/plots/sample_001100/susceptibility_comparison.png").is_file()
    assert (tmp_path/"evaluation/plots/sample_001100/tmi_comparison.png").is_file()
