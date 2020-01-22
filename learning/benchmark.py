#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Benchmark of learning methods."""
import json
import numpy as np
import time


class Benchmark:
    """A class to run the benchmark."""

    def __init__(self, dataset, noise=0.1):
        """Initialize the benchmark with the given data."""
        self._dataset = np.array(dataset)
        self._aggregate_metrics = {}
        self._noise = noise
        self._queries = self._create_queries(
            self._dataset, num_queries=min(1000, len(self._dataset))
        )
        self._triplets = {}  # (original, query, result) triplets

    def eval(self, algorithm, network, training_steps=1e6, name=None):
        """Benchmark this network and number of training steps."""
        if name is None:
            name = len(self._aggregate_metrics)

        data = np.array(self._dataset)

        training_start_time = time.time()
        convergence = algorithm.train(network, data, training_steps)
        training_end_time = time.time()

        retrievals = 0
        triplets = []
        for query, index in self.queries:
            network.input(query)
            network.stabilize()
            if all(data[index] == network.values):
                retrievals += 1

            triplets.append([data[index], query, network.values])

        metrics = {}
        metrics['training_time'] = training_end_time - training_start_time
        metrics['convergence'] = convergence
        metrics['retrieval'] = retrievals / len(self._queries)
        metrics['queries'] = len(self._queries)

        self._aggregate_metrics[name] = metrics
        self._triplets[name] = triplets

    def _create_queries(self, dataset, num_queries):
        data_indices = np.random.choice(len(dataset), num_queries, replace=False)
        mangled_data = list(self._apply_noise(dataset[i], self._noise)
                            for i in data_indices)
        return list(zip(mangled_data, data_indices))

    @property
    def queries(self):
        return self._queries

    @property
    def results(self):
        """Get the collected results."""
        return self._aggregate_metrics

    def __str__(self):
        """The string representation of this benchmark's results."""
        return "%s: %s" % (self.__class__, json.dumps(self.results, indent=4))

    @staticmethod
    def _apply_noise(datapoint, noise):
        num_noisy = int(np.ceil(len(datapoint) * noise))
        noisy_indices = np.random.choice(len(datapoint), num_noisy, replace=False)

        noisy_datapoint = np.array(datapoint).copy()
        noisy_datapoint[noisy_indices] = np.random.choice([1, -1], num_noisy, replace=True)
        return noisy_datapoint
