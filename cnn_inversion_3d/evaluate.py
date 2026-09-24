"""Evaluate every E01 test sample and create publication-style outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.analyze import analyze_prediction
from cnn_inversion_3d.dataset import load_magnetic_sample, read_manifest_paths
from evaluation.tmi_metrics import calculate_susceptibility_metrics

def _sections(volume: np.ndarray, center: tuple[int,int,int]):
    z,y,x=center
    return ((volume[z], "Horizontal X-Y", "X (m)", "Y (m)", (0,640,0,640)),
            (volume[:,y,:], "Vertical X-Z", "X (m)", "Depth (m)", (0,640,240,0)),
            (volume[:,:,x], "Vertical Y-Z", "Y (m)", "Depth (m)", (0,640,240,0)))

def save_susceptibility_comparison(true: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    occupied=np.argwhere(np.abs(true)>0)
    center=tuple(np.rint(occupied.mean(axis=0)).astype(int)) if occupied.size else tuple(np.array(true.shape)//2)
    vmax=max(float(np.max(np.abs(true))),float(np.max(np.abs(predicted))),1e-12)
    dmax=max(float(np.max(np.abs(predicted-true))),1e-12)
    figure,axes=plt.subplots(3,3,figsize=(14,12),constrained_layout=True)
    for row,(truth_section,label,xlabel,ylabel,extent) in enumerate(_sections(true,center)):
        pred_section=_sections(predicted,center)[row][0]; residual=pred_section-truth_section
        for col,(values,title,cmap,lo,hi) in enumerate(((truth_section,"True susceptibility","viridis",0,vmax),
                (pred_section,"Predicted susceptibility","viridis",0,vmax),(residual,"Prediction minus truth","RdBu_r",-dmax,dmax))):
            image=axes[row,col].imshow(values,origin="upper" if row else "lower",extent=extent,aspect="auto",cmap=cmap,vmin=lo,vmax=hi)
            axes[row,col].set_title(f"{title}\n{label}"); axes[row,col].set_xlabel(xlabel); axes[row,col].set_ylabel(ylabel)
            figure.colorbar(image,ax=axes[row,col],shrink=.78,label="Susceptibility (SI)")
    figure.suptitle("3D magnetic susceptibility reconstruction",fontsize=15,fontweight="bold")
    path.parent.mkdir(parents=True,exist_ok=True); figure.savefig(path,dpi=180); plt.close(figure)

def _write_csv(path: Path, rows: list[dict[str,object]]) -> None:
    if not rows:return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tmi-scale",type=float,default=100.0); parser.add_argument("--threshold-si",type=float,default=.001)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    consistency_root=args.output/"tmi_consistency"; consistency_root.mkdir(exist_ok=True)
    model=tf.keras.models.load_model(args.model,compile=False)
    paths=read_manifest_paths(dataset_directory=args.dataset.resolve(),manifest_name="test_manifest.csv")
    prediction_rows=[]; tmi_rows=[]; combined=[]
    for sample_path in paths:
        sample_id=sample_path.stem; tmi,true=load_magnetic_sample(sample_path)
        # Direct eager inference avoids Keras' compiled predict function and its
        # unnecessary XLA/PTX toolchain dependency for single-sample evaluation.
        normalized=tf.convert_to_tensor(tmi[None]/args.tmi_scale,dtype=tf.float32)
        predicted=np.asarray(model(normalized,training=False)[0],np.float32)
        true3=true[...,0]; predicted3=predicted[...,0]
        prediction_file=args.output/f"{sample_id}_prediction.npz"
        np.savez_compressed(prediction_file,tmi=tmi[...,0],true_susceptibility=true3,recovered_susceptibility=predicted3)
        save_susceptibility_comparison(true3,predicted3,args.output/f"{sample_id}_comparison.png")
        susceptibility_metrics=calculate_susceptibility_metrics(true3,predicted3,args.threshold_si)
        sample_dir=consistency_root/sample_id
        report=analyze_prediction(prediction_file,threshold_si=args.threshold_si,figure_path=sample_dir/"magnetic_comparison.png")
        (sample_dir/"tmi_consistency_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        prediction_row={"sample_id":sample_id,**susceptibility_metrics}; tmi_row={"sample_id":sample_id,**report["tmi_fit"]}
        prediction_rows.append(prediction_row); tmi_rows.append(tmi_row); combined.append({**prediction_row,**report["tmi_fit"]})
        print(f"Evaluated {sample_id}",flush=True)
    _write_csv(args.output/"prediction_metrics.csv",prediction_rows)
    _write_csv(args.output/"tmi_consistency_metrics.csv",tmi_rows)
    _write_csv(args.output/"combined_test_metrics.csv",combined)
    print(f"Prediction outputs written to: {args.output.resolve()}")

if __name__=="__main__":main()
