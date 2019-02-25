#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Implementation of Hebbian Learning for Hopfield Networks."""
import numpy as np


class HebbianLearning:
    def __init__(self, async=True):
        self._async = async

    def train(self, network, data, steps, convergence_threshold=1e-10):
        for _ in range(steps):
            data_t = np.transpose(data)
            # Symmetric matrix with matches, keeps connections symmetric.
            matches = np.dot(data_t, data)
            np.fill_diagonal(matches, 0)

            new_values = matches / len(data)
            delta_norm = np.linalg.norm(new_values - network._connections)

            network._connections = new_values
            if delta_norm < convergence_threshold:
                return _
        return False
