#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest

from networks import hopfield


@pytest.fixture
def hopfield_network():
    return hopfield.HopfieldNetwork(100)

def test_initialization(hopfield_network):
    for neuron in hopfield_network._neurons:
        assert neuron in (-1, 1)

    for i in range(hopfield_network.size):
        for j in range(hopfield_network.size):
            if i == j:
                assert hopfield_network._connections[i, j] == 0
            else:
                assert -1 < hopfield_network._connections[i, j] < 1
