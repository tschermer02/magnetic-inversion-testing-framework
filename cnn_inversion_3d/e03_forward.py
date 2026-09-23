"""Differentiable TensorFlow implementation of the E02 TMI operator."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from e01_magnetic.config import MagneticSurveyConfig
from forward_modeling.forward_model import inducing_field_direction, make_tensor_grid


class DifferentiableTMIForward(tf.keras.layers.Layer):
    """Point-dipole TMI operator matching ``TMIForwardModel`` in nT."""

    def __init__(self, receiver_chunk_size: int = 128, *, grid=None, receiver_xyz=None,
                 field_strength_nt=None, inclination_deg=None, declination_deg=None,
                 azimuth_deg=None, receiver_shape=None, **kwargs):
        super().__init__(trainable=False, **kwargs)
        if receiver_chunk_size < 1:
            raise ValueError("receiver_chunk_size must be positive.")
        survey = MagneticSurveyConfig()
        grid = grid or make_tensor_grid([0, 640, 0, 640, 0, 240], [10, 10], 10)
        receiver_xyz = survey.receiver_xyz if receiver_xyz is None else np.asarray(receiver_xyz)
        field_strength_nt = survey.field_strength_nt if field_strength_nt is None else field_strength_nt
        inclination_deg = survey.inclination_deg if inclination_deg is None else inclination_deg
        declination_deg = survey.declination_deg if declination_deg is None else declination_deg
        azimuth_deg = survey.azimuth_deg if azimuth_deg is None else azimuth_deg
        cells, volumes = grid.quadrature()
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.cell_xyz = tf.constant(cells.astype(np.float32))
        self.receiver_xyz = tf.constant(receiver_xyz.astype(np.float32))
        self.direction = tf.constant(inducing_field_direction(
            inclination_deg, declination_deg, azimuth_deg
        ).astype(np.float32))
        self.gamma = tf.constant((field_strength_nt * volumes / (4*np.pi)).astype(np.float32))
        self.n_receivers = int(receiver_xyz.shape[0])
        self.model_shape_zyx = (grid.z.size,grid.y.size,grid.x.size,1)
        self.receiver_shape = tuple(receiver_shape or ((81,81,1) if self.n_receivers==6561 else (self.n_receivers,1,1)))

    def call(self, susceptibility):
        values = tf.convert_to_tensor(susceptibility, tf.float32)
        tf.debugging.assert_equal(tf.shape(values)[1:], self.model_shape_zyx)
        models = tf.reshape(values, (tf.shape(values)[0], -1))  # z,y,x => x-fastest
        chunks = []
        for start in range(0, self.n_receivers, self.receiver_chunk_size):
            receivers = self.receiver_xyz[start:min(start+self.receiver_chunk_size,self.n_receivers)]
            delta = self.cell_xyz[None,:,:] - receivers[:,None,:]
            r2 = tf.reduce_sum(tf.square(delta), axis=2)
            dot = tf.einsum("rci,i->rc", delta, self.direction)
            kernel = self.gamma[None,:] * (3*tf.square(dot)/r2 - 1) / (r2*tf.sqrt(r2))
            chunks.append(tf.linalg.matmul(models, tf.stop_gradient(kernel), transpose_b=True))
        return tf.reshape(tf.concat(chunks, axis=1), (-1,*self.receiver_shape))

    def get_config(self):
        return {**super().get_config(), "receiver_chunk_size":self.receiver_chunk_size}
