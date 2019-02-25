#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from networks.hopfield import HopfieldNetwork
from networks.update import AsyncUpdate, SyncUpdate


@pytest.fixture
def neurons():
    return np.array([-1, 1, 1, 1])

@pytest.fixture
def connections():
    return np.array(
        [[ 0, +1, -1,  0],
         [+1,  0,  0, +1],
         [-1,  0,  0, +1],
         [ 0, +1, +1,  0]]
    )

@pytest.fixture
def expected_neurons():
    return np.array([-1, -1, 1, 1])

@pytest.fixture
def threshold():
    return 0.0

@pytest.fixture(params=list(range(4)))
def index(request):
    return request.param

def test_sync_update(
        neurons, connections, threshold, expected_neurons):
    expected_change = sum(neurons != expected_neurons)
    sync_rule = SyncUpdate()
    new_neurons, change = sync_rule(neurons, connections, threshold)

    assert change == expected_change
    assert (new_neurons == expected_neurons).all()

def test_async_update(
        neurons, connections, threshold, expected_neurons, index):
    # Will only update neurons[index]
    expected_change = int(neurons[index] != expected_neurons[index])
    async_rule = AsyncUpdate(lambda x: index)

    new_neurons, change = async_rule(neurons, connections, threshold)

    assert change == expected_change
    assert new_neurons[index] == expected_neurons[index]

    new_neurons, change = async_rule(neurons, connections, threshold)

def test_double_async_update_does_nothing(
        neurons, connections, threshold, index):
    async_rule = AsyncUpdate(lambda x: index)

    # Does some update
    async_rule(neurons, connections, threshold)

    # Will not change anything as it tries to update same index
    new_neurons, change = async_rule(neurons, connections, threshold)

    assert change == 0
