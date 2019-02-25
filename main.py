import numpy as np
import time
from learning.benchmark import Benchmark
from learning.hebbian import HebbianLearning
from networks.hopfield import HopfieldNetwork
from networks.update import SyncUpdate, AsyncUpdate


def benchmark(algorithm, dataset, update_rule):
    if dataset is None or len(dataset) is 0:
        return {}

    training_steps = [5]
    num_neurons = len(dataset[0])

    benchmark = Benchmark(algorithm, dataset)
    for steps in training_steps:
        benchmark.eval(
            HopfieldNetwork(num_neurons, update_rule=update_rule),
            training_steps=int(steps),
            name="{0:03} steps".format(steps))

    return benchmark

def dataset(num_datapoints, num_neurons):
    return np.sign(np.random.rand(num_datapoints, num_neurons) - 0.5)


if __name__ == "__main__":
    data = dataset(15, num_neurons=10000)
    print(data)
    print("\n\n\n\n\n")
    print(benchmark(HebbianLearning(), data, SyncUpdate()))

    print("Done!")
