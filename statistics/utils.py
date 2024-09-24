# calculate the five-number summary for the 'age' column, grouping by 'sex'
from typing import Tuple

import pandas as pd


def five_number_summary(values: pd.Series, show: bool = False) -> Tuple[float, float, float, float, float]:
    """
    Show the five-number summary of a given pandas Series.
    :param values: The pandas Series to show the five-number summary.
    :param show: Whether we should print the values or not.
    :return: A tuple with the five-number summary values: min, Q1, median, Q3, max.
    """
    min, q1, median, q3, max = values.min(), values.quantile(0.25), values.median(), values.quantile(0.75), values.max()
    if show:
        print(f"\tMinimum: {min}.")
        print(f"\t1st quartile (Q1): {q1}.")
        print(f"\tMedian (Q2): {median}.")
        print(f"\t3rd quartile (Q3): {q3}.")
        print(f"\tMaximum: {max}.")
    return min, q1, median, q3, max


