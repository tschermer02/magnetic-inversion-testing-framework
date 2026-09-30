import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from cnn_inversion_3d.e01_core_loss import _per_sample_e01_terms,build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import rational_soft_occupancy
from cnn_inversion_3d.e06_callbacks import validation_metrics
from cnn_inversion_3d.e06_config import E06SuiteConfig,VARIANTS
from cnn_inversion_3d.e06_runner import build_model,loss_config,train_variant
from cnn_inversion_3d.e05_config import FIRST_STAGE,dependent_variants
from cnn_inversion_3d.e05_runner import loss_config as e05_loss_config

def test_variant_coefficients():
    assert [(VARIANTS[x].body_fraction,VARIANTS[x].occupancy_mode) for x in VARIANTS]==[(.5,"legacy_threshold_sigmoid"),(.75,"legacy_threshold_sigmoid"),(.5,"rational"),(.75,"rational")]
    for variant in VARIANTS.values():
        cfg=loss_config(variant);assert cfg.lambda_susceptibility==100 and cfg.lambda_tmi==1e-4 and cfg.lambda_tversky==.1

def test_body_background_arithmetic_and_e06a_e05e_match():
    truth=np.zeros((1,24,64,64,1),np.float32);truth[:,0,0,0]=1
    prediction=np.full_like(truth,2);prediction[:,0,0,0]=0
    truth=tf.constant(truth);prediction=tf.constant(prediction);_,weights=build_e01_sensitivity_weights()
    a=_per_sample_e01_terms(truth,prediction,weights,loss_config(VARIANTS["e06a"]))[0]
    b=_per_sample_e01_terms(truth,prediction,weights,loss_config(VARIANTS["e06b"]))[0]
    assert np.isclose(float(a),2.5) and np.isclose(float(b),1.75)
    old=e05_loss_config(dependent_variants(FIRST_STAGE["e05c"])["e05e"]);new=loss_config(VARIANTS["e06a"])
    for name in ("lambda_susceptibility","body_fraction","lambda_depth","lambda_sensitivity","lambda_amplitude","lambda_tmi","lambda_tversky","tversky_alpha","tversky_beta","occupancy_threshold","occupancy_sharpness","occupancy_mode"):
        assert getattr(new,name)==getattr(old,name)

def test_rational_mapping_values_monotonic_and_finite_gradient():
    x=tf.Variable([0.,.001,.1])
    with tf.GradientTape() as tape:
        y=rational_soft_occupancy(x,tau_si=.001);total=tf.reduce_sum(y)
    gradient=tape.gradient(total,x).numpy()
    assert np.allclose(y.numpy()[:2],[0,.5]) and np.all(np.diff(y.numpy())>0) and np.all(np.isfinite(gradient)) and np.all(gradient>0)

class _Identity:
    def __init__(self,pred):
        value=tf.constant(pred,tf.float32)
        self.inversion_model=lambda x,training=False:tf.repeat(value,tf.shape(x)[0],axis=0)

def test_balanced_validation_includes_background_and_canonical_iou():
    truth=np.zeros((1,2,1,1,1),np.float32);truth[:,0]=.1;pred=truth.copy();pred[:,1]=.1
    values=validation_metrics(_Identity(pred),tf.data.Dataset.from_tensor_slices((np.zeros((1,1)),truth)).batch(1),.001)
    assert np.isclose(values["validation_balanced_mae"],.5) and np.isclose(values["iou"],.5)

def test_incompatible_resume_rejected(tmp_path):
    out=tmp_path/"e06a";out.mkdir(parents=True);(out/"run_config.json").write_text(json.dumps({"compatibility_hash":"wrong"}))
    with pytest.raises(ValueError,match="Incompatible"):train_variant(E06SuiteConfig(dataset=Path("datasets/E01_soft_tversky_full")),VARIANTS["e06a"],tmp_path,resume=True)

def test_continuous_prediction_reaches_forward_operator():
    config=E06SuiteConfig(base_filters=1);model=build_model(config,VARIANTS["e06a"]);x=tf.zeros((1,81,81,1));prediction=model.inversion_model(x)
    captured=model.forward_operator(prediction)
    assert prediction.shape==(1,24,64,64,1) and captured.shape==(1,81,81,1) and np.all(np.isfinite(captured.numpy()))

def test_train_save_reload_predict_analyze_smoke(tmp_path):
    sample=Path("datasets/E01_soft_tversky_full/samples/sample_001100.npz")
    if not sample.is_file():pytest.skip("E01 test data unavailable")
    with np.load(sample) as saved:
        tmi=tf.constant(saved["tmi"][None,...,None]/100,tf.float32)
        truth=tf.constant(saved["susceptibility"][None,...,None],tf.float32)
    config=E06SuiteConfig(base_filters=1);variant=VARIANTS["e06a"];model=build_model(config,variant)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False);model(tmi,training=False);model.train_step((tmi,truth))
    training=tmp_path/"training";training.mkdir();model.save_weights(training/"primary.weights.h5")
    reloaded=build_model(config,variant);reloaded(tf.zeros((1,81,81,1)),training=False);reloaded.load_weights(training/"primary.weights.h5")
    assert np.all(np.isfinite(reloaded.inversion_model(tmi).numpy()))
    from evaluation.e05_evaluate import evaluate_variant
    rows,summary=evaluate_variant(config,variant,training,tmp_path/"evaluation",limit=1,plots="none",
        checkpoint_name="primary.weights.h5",checkpoint_label="primary",model_builder=build_model)
    assert len(rows)==1 and summary["checkpoint"]["label"]=="primary" and np.isfinite(rows[0]["tmi_rmse_nt"])
