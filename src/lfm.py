"""lfm.py

Latent factor model.
"""

import operator
import random
import typing

import numpy as np


random.seed(0)
np.random.seed(0)


class LFM():
    """Latent factor model for recommendation.

    Attributes:
        all_user, Dict[int, Set[int]]. {uid: {iid}}.
        all_iid, Set[int]. All iid.
        X, np.ndarray[|U|, D]. User latent factor.
        W, np.ndarray[|I|, D]. Item latent factor.
    """
    def __init__(self):
        """Initialization."""
        self.all_user = None
        self.all_iid = None
        self.X = None
        self.W = None

    def train(self, all_user: typing.Dict[int, typing.Set[int]],
              dimension: int, ratio: float = 1.0, epochs: int = 80,
              lr: float = 1e-2, lambda_: float = 5e-4):
        """Training.

        Args:
            all_user, Dict[int, Set[int]]. {uid: {iid}}
            dimension, int. Latent factor dimension.
            ratio, float. Negative samples / positive samples
            epochs, int. Training epochs.
            lambda_, float. Regularization coefficient.
            lr, float. Learning rate
        """
        # Initialization.
        self.all_user = all_user
        self.all_iid = {
            iid for uid, bought in self.all_user.items() for iid in bought}
        self.X = np.random.rand(len(self.all_user), dimension)
        self.W = np.random.rand(len(self.all_iid), dimension)

        # Start training.
        for _ in range(epochs):
            data = self._sampleData(ratio)
            for (uid, iid), y in data.items():
                uid -= 1  # starts with 0
                iid -= 1  # starts with 0
                self.X[uid] -= lr * ((np.dot(self.W[iid], self.X[uid]) - y)
                                     * self.W[iid] + lambda_ * self.X[uid])
                self.W[iid] -= lr * ((np.dot(self.W[iid], self.X[uid]) - y)
                                     * self.X[uid] + lambda_ * self.W[iid])
            lr *= 0.9  # Learning rate decay

    def _sampleData(self, ratio: float = 1.0) -> typing.Dict[typing.Tuple[int],
                                                             int]:
        """Sample negative data for training.

        Args:
            ratio, float. Negative samples / positive samples

        Returns:
            data, Dict[Tuple[int], int]. {(uid, iid): y}
        """
        data = {}
        for uid, bought in self.all_user.items():
            for iid in bought:
                data[(uid, iid)] = 1
            not_bought = random.choices(list(self.all_iid - bought),
                                        k=int(len(bought) * ratio))
            for iid in not_bought:
                data[(uid, iid)] = 0
        return data

    def predict(self, uid: int, N: int) -> (typing.List[int],
                                            typing.List[float]):
        """Make prediction for the given user.

        Args:
            uid, int. User ID.
            N, int: Recommendation list length.

        Returns:
            prediction, List[int]. Predicted item ID.
            interest, List[float]. User interest to the given item ID.
        """
        # Dict[int, float]. {iid: interest}.
        interest_dict = {
            iid: np.dot(self.W[iid - 1], self.X[uid - 1])
            for iid in self.all_iid if iid not in self.all_user[uid]}

        # Make prediction.
        prediction = []
        interest = []
        for iid, interest_value in sorted(
                interest_dict.items(), key=operator.itemgetter(1),
                reverse=True)[:N]:
            prediction.append(iid)
            interest.append(interest_value)
        return prediction, interest


def main():
    all_user = {1: {1, 2, 4}, 2: {1, 3}, 3: {2, 5}, 4: {3, 4, 5}}
    model = LFM()
    model.train(all_user, dimension=256)
    prediction, interest = model.predict(uid=1, N=2)
    print(prediction, interest)
    # Output: [3, 5] [0.2477385428185519, 0.22120916570428018]


if __name__ == '__main__':
    main()
