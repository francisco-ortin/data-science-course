import pandas as pd
import random
from sklearn.linear_model import LogisticRegression


class AlwaysTrue:
    """This class always predicts the positive class"""

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts always the positive class"""
        return pd.Series([1] * len(X_test))

    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        """Returns the probability of the positive class for each sample"""
        return self.predict(X_test)


    def __str__(self):
        return "AlwaysTrueModel"


class AlwaysFalse:
    """This class always predicts the negative class"""

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts always the negative class"""
        return pd.Series([0] * len(X_test))

    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        """Returns the probability of the positive class for each sample"""
        return self.predict(X_test)

    def __str__(self):
        return "AlwaysFalseModel"

class RandomModel:
    """This class predicts randomly the positive or negative class"""

    def __init__(self, random_state: int):
        random.seed(random_state)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts randomly the positive or negative class"""
        return pd.Series([random.randint(0, 1) for _ in range(len(X_test))])

    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        """Returns a random probability for each sample"""
        return pd.Series([random.random() for _ in range(len(X_test))])

    def __str__(self):
        return "RandomModel"

class ThresholdModel:
    """This class predicts the positive class if the probability is higher than a threshold"""

    def __init__(self, model: LogisticRegression, threshold: float):
        self.threshold = threshold
        self.model = model

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Predicts the positive class if the probability of the model is higher or equal than the threshold"""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        return pd.Series([1 if prob >= self.threshold else 0 for prob in y_pred_proba])

    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        """Returns the probability of the positive class for each sample"""
        return self.model.predict_proba(X_test)[:, 1]

    def __str__(self):
        return f"LogisticRegressionModel(threshold={self.threshold})"
