import numpy
from collections import Counter

import numpy as np
import pandas
import pandas as pd


class MyKNN:

    def __init__(self, k=2):
        self.train = None
        self.target = None
        self.k = k


    def fit(self, A, b):
        if not isinstance(A, numpy.ndarray):
            self.train = A.to_numpy()
        else:
            self.train = A

        if not isinstance(b, numpy.ndarray):
            self.target = b.to_numpy()
        else:
            self.target = b


    def predict(self, P, axis=0):

        if axis == 0:
            # Euclidean distance
            # Expand summation to polynomial like structure
            # T^2 + P^2 - 2PT^T
            T = numpy.sum(self.train**2, axis=1)
            P2 = numpy.sum(P**2, axis=1)
            TP = P @ self.train.T
            dist_squares = (T[numpy.newaxis, :] + P2[:, numpy.newaxis]) - 2 * TP

            # Found some negative values, maximum between 0 to adjust
            dist_squares = numpy.maximum(dist_squares, 0)
            distances = numpy.sqrt(dist_squares)
        elif axis == 1:
            # Manhattan distance
            diff = numpy.abs(self.train[numpy.newaxis, :, :] - P[:, numpy.newaxis, :,])
            distances = numpy.sum(diff, axis=2)
        else:
            return ValueError("Axis value incorrect")


        smallest_k = numpy.argpartition(distances, self.k)[:, :self.k]  # Partition distances to get the leading k values as the closest neighbors
        k_labels = self.target[smallest_k]  # Index the target set for the corresponding labels

        # Identify unique labels for target
        unique_labels = numpy.unique(self.target)

        # Stretch 2d k_labels to broadcast equality function and sum across the second axis
        counts = numpy.equal(k_labels[:, :, numpy.newaxis], unique_labels).sum(axis=1)

        # Use the argmax() to get the index of the majority and pass to unique_labels
        predictions = unique_labels[numpy.argmax(counts, axis=1)]

        return numpy.array(predictions)


