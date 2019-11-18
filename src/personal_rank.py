"""personal_rank.py

PersonalRank graph model.
"""

import typing

import numpy as np


class PersonalRank():
    """PersonalRank model for recommendation.

    Attributes:
        all_user, Dict[int, Set[int]]. {uid: {iid}}
        alpha, float. (1 - alpha) is the probability to restart.
        A, np.ndarray[|U| + |I|, |U| + |I|]. Graph transition matrix.
        p, np.ndarray(|U| + |I|). State vector.
    """
    def __init__(self):
        """Initialization."""
        self.all_user = None
        self.alpha = None
        self.A = None
        self.p = None

    def train(self, all_user: typing.Dict[int, typing.Set[int]],
              alpha: float = 0.9):
        """Training.

        Args:
            all_user, Dict[int, Set[int]]. {uid: {iid}}
            alpha, float. (1 - alpha) is the probability to restart.
        """
        # Initialization.
        self.all_user = all_user
        self.alpha = alpha
        all_iid = {
            iid for uid, bought in self.all_user.items() for iid in bought}
        self.A = np.zeros((len(self.all_user) + len(all_iid),
                           len(self.all_user) + len(all_iid)))
        self.p = np.zeros(len(self.all_user) + len(all_iid))

        # Fill in matrix A.
        for uid, bought in self.all_user.items():
            for iid in bought:
                self.A[uid - 1][len(self.all_user) + iid - 1] = 1
                self.A[len(self.all_user) + iid - 1][uid - 1] = 1
        self.A /= np.sum(self.A, axis=0, keepdims=True)

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
        # Initial state.
        self.p[uid - 1] = 1
        # Solve the equation.
        p = np.linalg.solve(1 - self.alpha * self.A,
                            (1 - self.alpha) * self.p)

        # Make prediction.
        p = p[len(self.all_user):]
        prediction = [
            iid + 1 for iid in np.argsort(p)[::-1].tolist()
            if (iid + 1) not in self.all_user[uid]][:N]
        interest = [p[prediction[i] - 1] for i in range(N)]
        return prediction, interest


def main():
    all_user = {1: {1, 2, 4}, 2: {1, 3}, 3: {2, 5}, 4: {3, 4, 5}}
    model = PersonalRank()
    model.train(all_user)
    prediction, interest = model.predict(uid=1, N=2)
    print(prediction, interest)
    # Output: [3, 5] [0.17565664098102587, -0.0371106876202577]


if __name__ == '__main__':
    main()
