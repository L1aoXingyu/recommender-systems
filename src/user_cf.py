"""user_cf.py

UserCF model.
"""


import collections
import itertools
import math
import operator
import typing


class UserCF():
    """User CF model for recommendation.

    Attributes:
        user_similarity, Dict[int, Dict[int, float]]. {uid1: {uid2: similarity}}.
        all_user, Dict[int, Set[int]]. {uid: {iid}}
    """
    def __init__(self):
        self.user_similarity = None
        self.all_user = None

    def train(self, all_user: typing.Dict[int, typing.Set[int]]):
        """Compute user similarity."""
        # Save user data.
        self.all_user = all_user

        # Build item-user inverse table.
        all_item = collections.defaultdict(set)
        for uid, bought in all_user.items():
            for iid in bought:
                all_item[iid].add(uid)

        # Co-rated items betweens users.
        # C, Dict[int, Dict[int, int]]. {uid1: {uid2: |f(u1) \cap f(u2)|}}
        C = collections.defaultdict(lambda: collections.defaultdict(int))
        for iid, bought_by in all_item.items():
            for uid1, uid2 in itertools.permutations(bought_by, 2):
                C[uid1][uid2] += 1

        # Compute final similarity.
        self.user_similarity = {
            uid1: {
                uid2: C[uid1][uid2] / math.sqrt(
                    len(all_user[uid1]) * len(all_user[uid2]))
                for uid2 in C[uid1]}
            for uid1 in C}

    def predict(self, uid: int, N: int, top_k: int) -> (
            typing.List[int], typing.List[float]):
        """Make prediction for the given user.

        Args:
            uid, int. User ID.
            N, int: Recommendation list length.
            top_k, int: Consider top-k similar users.
        """
        # Dict[int, float]. {iid: interest}.
        interest_dict = collections.defaultdict(float)
        for uid2, similarity_value in sorted(
                self.user_similarity[uid].items(), key=operator.itemgetter(1),
                reverse=True)[:top_k]:
            for iid in self.all_user[uid2]:
                if iid not in self.all_user[uid]:
                    interest_dict[iid] += similarity_value

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
    model = UserCF()
    model.train(all_user)
    prediction, interest = model.predict(uid=1, N=2, top_k=3)
    print(prediction, interest)
    # Output: [3, 5] [0.7415816237971964, 0.7415816237971964]


if __name__ == '__main__':
    main()
