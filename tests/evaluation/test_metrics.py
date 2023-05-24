import numpy as np
import pytest

from src.evaluation import metrics


@pytest.fixture
def example_actual_labels():
    return np.array([2, 1, 0, 0, 1, 0, 1, 2, 2, 0, 2])


@pytest.fixture
def example_cluster_labels():
    return np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])


def test_pairwise_f1(example_actual_labels, example_cluster_labels):
    actual_f1 = metrics.pairwise_f1(example_actual_labels, example_cluster_labels)
    assert 1 / 3 == actual_f1


def test_pairwise_precision(example_actual_labels, example_cluster_labels):
    actual_precision = metrics.pairwise_precision(
        example_actual_labels, example_cluster_labels
    )
    assert 1 / 3 == actual_precision


def test_pairwise_recall(example_actual_labels, example_cluster_labels):
    actual_recall = metrics.pairwise_recall(
        example_actual_labels, example_cluster_labels
    )
    assert 1 / 3 == actual_recall


def test__true_positive(example_actual_labels, example_cluster_labels):
    actual_tp = metrics._true_positive(example_actual_labels, example_cluster_labels)
    assert 5 == actual_tp


def test__false_positive(example_actual_labels, example_cluster_labels):
    actual_fp = metrics._false_positive(example_actual_labels, example_cluster_labels)
    assert 10 == actual_fp


def test__false_negative(example_actual_labels, example_cluster_labels):
    actual_fn = metrics._false_negative(example_actual_labels, example_cluster_labels)
    assert 10 == actual_fn
