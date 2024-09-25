import numpy as np


class Model:
    def __init__(self, distribution: str, mean: float, std: float) -> None:
        self.distribution = distribution
        self.mean = mean
        self.std = std
        if distribution.lower() not in ["normal", "uniform"]:
            raise ValueError(f"Invalid distribution: {distribution}.")

    def train(self, data: np.ndarray) -> None:
        """
        Train the model with the given data.
        :param data: The data to train the model, including the target (last column).
        """
        pass

    def accuracy(self, data: np.ndarray) -> float:
        """
        Evaluate the model with the given data.
        :param data: The data to predict, including the target (last column).
        :return: The accuracy of the model.
        """
        if self.distribution.lower() == "normal":
            # generate data_size random values from a normal distribution (mean and std)
            return np.random.normal(self.mean, self.std, 1)[0]
        elif self.distribution.lower() == "uniform":
            # generate data_size random values from a uniform distribution (mean and std)
            return np.random.uniform(self.mean, self.std, 1)[0]
        assert False, f"Invalid distribution: {self.distribution}."
