#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Implementation of a Hopfield Network."""
import abc
import numpy as np


class RandomIndexGenerator:
    """A random index generator."""

    def __call__(self, arr):
        return np.random.randint(len(arr))

class SequentialIndexGenerator:
    """A serial index generator."""

    def __init__(self):
        """Initialize the index generator."""
        self._index = 0

    def __call__(self, arr):
        return self._index % len(arr)


class UpdateRule(abc.ABC):
    """A base UpdateRule class."""

    @abc.abstractmethod
    def __call__(self, neurons, connections, threshold):
        """
        Computes the new values.

        Returns a tuple with (new_values, neurons_changed).
        Might edit the neurons in place and return the same instance
        or a new one.
        """
        raise NotImplementedError()


class AsyncUpdate(UpdateRule):
    """An update rule in which neurons are updated asynchronously."""

    def __init__(self, index_generator=SequentialIndexGenerator()):
        """Initialize the AsyncUpdate rule."""
        self._index_generator = index_generator

    def __call__(self, neurons, connections, threshold):
        """Update a random neuron."""
        index = self._index_generator(neurons)
        stored_value = neurons[index]

        value = np.dot(neurons, connections[index])
        neurons[index] = (value > threshold) * 2 - 1

        return neurons, int(neurons[index] != stored_value)


class SyncUpdate(UpdateRule):
    """An update rule in which all neurons are updated all at once."""

    def __call__(self, neurons, connections, threshold):
        """Update all neurons."""
        values = np.dot(connections, neurons)
        new_neurons = (values > threshold) * 2 - 1

        return new_neurons, sum(neurons != new_neurons)
