"""Run the E01 TMI-to-susceptibility model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.dataset import load_magnetic_sample

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--sample",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tmi-scale",type=float,default=100.0); args=parser.parse_args()
    tmi,true=load_magnetic_sample(args.sample); model=tf.keras.models.load_model(args.model,compile=False)
    normalized=tf.convert_to_tensor(tmi[None]/args.tmi_scale,dtype=tf.float32)
    recovered=np.asarray(model(normalized,training=False)[0],np.float32)
    np.savez_compressed(args.output,tmi=tmi[...,0],true_susceptibility=true[...,0],recovered_susceptibility=recovered[...,0])

if __name__=="__main__": main()
