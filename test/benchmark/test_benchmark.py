#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np

from learning import benchmark


def test_noise():
    point = np.array([1] * 100)
    noisy_point = np.array(benchmark.Benchmark._apply_noise(point, noise=0.5))
    assert any(point != noisy_point)
