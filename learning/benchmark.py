#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Benchmark of learning methods."""
import json
import numpy as np
import time


class Benchmark:
    """A class to run the benchmark."""

    def __init__(self, algorithm, dataset, mangling=1):
        """Initialize the benchmark with the given algorithm and data."""
        self._dataset = np.array(dataset)
        self._algorithm = algorithm
        self._aggregate_metrics = {}
        self._mangling = mangling
        self._queries = self._create_queries(
            self._dataset, num_queries=1000
        )

    def eval(self, network, training_steps=1e6, name=None):
        """Benchmark this network and number of training steps."""
        if name is None:
            name = len(self._aggregate_metrics)

        data = np.array(self._dataset)

        training_start_time = time.time()
        convergence = self._algorithm.train(network, data, training_steps)
        training_end_time = time.time()

        retrievals = 0
        for query, index in self._queries:
            network.input(query)
            network.stabilize()
            if all(data[index] == network.values):
                retrievals += 1

            print(np.array([query, network.values, data[index]]))

        metrics = {}
        metrics["training_time"] = training_end_time - training_start_time
        metrics["convergence"] = convergence
        metrics["retrieval"] = retrievals / len(self._queries)
        metrics["queries"] = len(self._queries)

        self._aggregate_metrics[name] = metrics

    def _create_queries(self, dataset, num_queries):
        included = set((tuple(x) for x in dataset))
        queries = set()
        queries_with_indices = set()
        attempts = 0
        limit = min(100, 5 * len(dataset))
        while len(queries) < num_queries and attempts < limit:
            attempts += 1
            index = np.random.randint(len(dataset))
            base = np.array(dataset[index])
            possible_query = Benchmark._mangle(base, self._mangling)
            if possible_query not in queries:
                differences = dataset - possible_query
                distance = np.sum(differences != 0, 1)
                if sum(distance <= self._mangling) == 1:
                    queries.add(possible_query)
                    queries_with_indices.add((possible_query, index))

        return list(queries_with_indices)

    @property
    def results(self):
        """Get the collected results."""
        return self._aggregate_metrics

    def __str__(self):
        """The string representation of this benchmark's results."""
        return "%s: %s" % (self.__class__, json.dumps(self.results, indent=4))

    @staticmethod
    def _mangle(datapoint, mangling):
        indices_for_mangling = np.random.choice(len(datapoint), replace=False)
        datapoint[indices_for_mangling] *= -1
        return tuple(datapoint)
