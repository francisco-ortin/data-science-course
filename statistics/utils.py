import numpy as np
import pandas as pd
from scipy.stats import norm, t


def five_number_summary(values: pd.Series | np.ndarray, show: bool = False) -> tuple[float, float, float, float, float]:
    """
    Show the five-number summary of a given pandas Series or NumPy ndarray.
    :param values: The pandas Series to show the five-number summary.
    :param show: Whether we should print the values or not.
    :return: A tuple with the five-number summary values: min, Q1, median, Q3, max.
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)
    values = pd.Series(values)
    min, q1, median, q3, max = values.min(), values.quantile(0.25), values.median(), values.quantile(0.75), values.max()
    if show:
        print(f"\tMinimum: {min}.")
        print(f"\t1st quartile (Q1): {q1}.")
        print(f"\tMedian (Q2): {median}.")
        print(f"\t3rd quartile (Q3): {q3}.")
        print(f"\tMaximum: {max}.")
    return min, q1, median, q3, max


def confidence_interval(sample: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Compute the confidence interval for a given sample, with two distributions: normal and student.
    :param sample: The sample to compute the confidence interval.
    :param confidence_level: The confidence level (default is 0.95).
    :return: A tuple with the lower and upper bounds of the confidence interval.
    """
    alpha = 1 - confidence_level
    mean = np.mean(sample)
    std = np.std(sample)
    n = len(sample)
    if n > 30:
        # normal distribution
        z = norm.ppf(1 - alpha / 2)
        return mean - z * std / np.sqrt(n), mean + z * std / np.sqrt(n)
    else:
        # student distribution
        t_value = t.ppf(1 - alpha / 2, n - 1)
        return mean - t_value * std / np.sqrt(n), mean + t_value * std / np.sqrt(n)


def do_intervals_overlap(interval1: tuple[float, float], interval2: tuple[float, float]) -> bool:
    """
    Check if two intervals overlap.
    :param interval1: The first interval.
    :param interval2: The second interval.
    :return: Whether the intervals overlap or not.
    """
    return interval1[1] >= interval2[0] and interval1[0] <= interval2[1]


def cohen_d(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """
    Compute the Cohen's d effect size between two samples.
    :param x: first sample
    :param y: second sample
    :return: value of Cohen's d
    """
    n_x = len(x)
    n_y = len(y)
    dof = n_x + n_y - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((n_x - 1) * np.std(x, ddof=1) ** 2 + (n_y - 1) * np.std(y, ddof=1) ** 2) / dof)
