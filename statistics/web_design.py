import numpy as np


class WebDesign:
    def __init__(self, name: str, mean: float, std: float) -> None:
        self.name = name
        self.mean = mean
        self.std = std

    def measure_network_traffic(self) -> float:
        """
        Measure the network traffic.
        :return: The network traffic measured in GBs per minute.
        """
        return np.random.normal(self.mean, self.std, 1)[0]
