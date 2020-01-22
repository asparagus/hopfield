#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import time

from PIL import Image

from learning import benchmark
from learning import hebbian
from networks import hopfield
from networks import update


SAMPLE_DIRECTORY = './samples'


def image_to_neurons(image):
    """Convert image to neuron representation."""
    buff = image.tobytes()
    byte_arr = np.frombuffer(buff, dtype='uint8')
    bits = np.unpackbits(byte_arr)
    return bits.astype('int8') * 2 - 1


def neurons_to_image(neurons, size):
    """Convert neuron representation to image."""
    binary_array = np.array(neurons) == 1
    byte_arr = np.packbits(binary_array.astype('uint8'))
    buff = byte_arr.tobytes()
    return Image.frombytes('1', size, buff)


def concatenate(images):
    """Concatenate images horizontally."""
    total_width = sum(img.width for img in images)
    dst = Image.new('1', (total_width, images[0].height))
    for i, img in enumerate(images):
        dst.paste(img, (i * img.width, 0))
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiments with a Hopfield Network')
    parser.add_argument('--directory', type=str,
                        help='Directory to obtain the sample images from',
                        default=SAMPLE_DIRECTORY)
    parser.add_argument('--noise', type=float,
                        help='Fraction of bits randomized for evaluation',
                        default=0.1)
    parser.add_argument('--output', type=str,
                        help='Output directory',
                        default='output')

    args = parser.parse_args()
    if args.directory == SAMPLE_DIRECTORY:
        print('Running script on samples directory\n'
              'To specify your own images use the --directory argument\n'
              'To view all available options use -h\n\n')

    image_files = os.listdir(args.directory)
    print('Found %i image files' % len(image_files))

    images = [Image.open(os.path.join(args.directory, f)) for f in image_files]

    min_width = min(min(img.size[0] for img in images), 128)
    min_height = min(min(img.size[1] for img in images), 128)

    min_size = min_width, min_height
    print('Transforming images to %ix%i' % min_size)
    print('...')

    quantized_images = [img.resize(min_size).convert('1') for img in images]

    num_neurons = min_width * min_height

    data = np.vstack([image_to_neurons(img) for img in quantized_images])
    print('Data preparation ready.')

    algorithm = hebbian.HebbianLearning()
    network = hopfield.HopfieldNetwork(num_neurons, update_rule=update.SyncUpdate())

    b = benchmark.Benchmark(data, noise=args.noise)
    b.eval(
        algorithm,
        network,
        training_steps=1
    )

    print(b)  # Print statistics

    for _, triplets in b._triplets.items():
        for i, triplet in enumerate(triplets):
            triplet_img = concatenate([neurons_to_image(img, min_size)
                                       for img in triplet])
            output_path = os.path.join(args.output, '%i.jpg' % i)
            triplet_img.save(output_path)

    print('Output saved to %s' % args.output)
