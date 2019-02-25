#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Implementation of a Hopfield Network."""
import numpy as np
from networks.update import SyncUpdate


class HopfieldNetwork:
    """A Hopfield network."""

    def __init__(self, size, threshold=0.0, update_rule=SyncUpdate()):
        """Initialize a network of the given size."""
        self._neurons = self._random_neuron_values(size)
        self._connections = self._random_connections(size)
        self._threshold = threshold
        self._update_rule = update_rule

    def update(self):
        """Updates the current neuron values to reach an energy minimum."""
        self._neurons, change = self._update_rule(
            self._neurons, self._connections, self._threshold
        )
        return change

    def stabilize(self, max_tries=1e6):
        """Updates the current configuration until a stable state is reached."""
        count = 0
        while count < max_tries and self.update():
            count += 1
        return self._neurons[:]

    def input(self, values):
        """Set the input values."""
        self._neurons[:] = values

    @property
    def size(self):
        """The size of the network (i.e. number of neurons)."""
        return len(self._neurons)

    @property
    def threshold(self):
        """The threshold for neuron values to shift between 1/-1."""
        return self._threshold

    @property
    def values(self):
        return self._neurons

    def _random_neuron_values(self, size):
        """Helper function to create random neuron values in {-1, 1}."""
        return  np.sign(np.random.rand(size) - 0.5)

    def _random_connections(self, size):
        """
        Helper function to create connection values in (-1, 1).

        Values in the diagonal will all be zero, signaling that neurons
        are not connected to themselves.
        """
        values = np.random.rand(size, size) * 2 - 1
        np.fill_diagonal(values, 0)
        return values

    def _sign(self, value):
        """The sign to be used when displaying this network."""
        return "+" if value == 1 else "-"

    def __str__(self):
        """The string representation of this network."""
        return "[%s]" % " ".join(
            self._sign(value) for value in self._neurons
        )
