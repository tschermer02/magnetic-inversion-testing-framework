"""Differentiable TensorFlow transcription of the fixed E01 TMI operator."""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from e01_magnetic.config import MagneticSurveyConfig
from forward_modeling.forward_model import inducing_field_direction,make_tensor_grid

class E04TMIForward(tf.keras.layers.Layer):
    """Map ``[batch,24,64,64,1]`` susceptibility SI to physical TMI nT."""
    def __init__(self,receiver_chunk_size:int=128,**kwargs):
        super().__init__(trainable=False,**kwargs)
        if receiver_chunk_size<1: raise ValueError("receiver_chunk_size must be positive")
        survey=MagneticSurveyConfig(); grid=make_tensor_grid([0,640,0,640,0,240],[10,10],10)
        cells,volumes=grid.quadrature()
        self.receiver_chunk_size=int(receiver_chunk_size)
        self.cell_xyz=tf.constant(cells.astype(np.float32))
        self.receiver_xyz=tf.constant(survey.receiver_xyz.astype(np.float32))
        self.direction=tf.constant(inducing_field_direction(survey.inclination_deg,
            survey.declination_deg,survey.azimuth_deg).astype(np.float32))
        self.gamma=tf.constant((survey.field_strength_nt*volumes/(4*np.pi)).astype(np.float32))
        self.model_shape=(24,64,64,1); self.n_receivers=81*81
    def call(self,susceptibility):
        values=tf.convert_to_tensor(susceptibility,tf.float32)
        tf.debugging.assert_equal(tf.shape(values)[1:],self.model_shape)
        models=tf.reshape(values,(tf.shape(values)[0],-1))
        responses=[]
        for start in range(0,self.n_receivers,self.receiver_chunk_size):
            receivers=self.receiver_xyz[start:min(start+self.receiver_chunk_size,self.n_receivers)]
            delta=self.cell_xyz[None,:,:]-receivers[:,None,:]
            r2=tf.reduce_sum(tf.square(delta),axis=2)
            dot=tf.einsum("rci,i->rc",delta,self.direction)
            kernel=self.gamma[None,:]*(3*tf.square(dot)/r2-1)/(r2*tf.sqrt(r2))
            responses.append(tf.linalg.matmul(models,tf.stop_gradient(kernel),transpose_b=True))
        return tf.reshape(tf.concat(responses,axis=1),(-1,81,81,1))
    def get_config(self): return {**super().get_config(),"receiver_chunk_size":self.receiver_chunk_size}
