#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Implementation of Hebbian Learning for Hopfield Networks."""
import numpy as np


class HebbianLearning:
    def __init__(self):
        pass

    def train(self, network, data, steps, convergence_threshold=1e-10):
        # Steps are ignored in this implementation
        data_t = np.transpose(data)
        # Symmetric matrix with matches, keeps connections symmetric.
        matches = np.dot(data_t, data)
        np.fill_diagonal(matches, 0)

        network._connections = matches / len(data)
        return 1  # 'Convergence' on first step.
