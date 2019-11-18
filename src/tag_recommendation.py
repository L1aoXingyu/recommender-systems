"""tag_recommendation.py

Recommendation based on tag information.
"""


import collections
import operator

import typing


class TagRecommendation():
    """Recommendation based on tag information.

    Attributes:
        all_user, Dict[int, Set[int]]. {uid: {iid}}
        all_iid, Set[int]. All iid
        tag2user, Dict[str, Dict[int, int]]. {tag: {uid: count}}
        tag2item, Dict[str, Dict[int, int]]. {tag: {iid: count}}
    """
    def __init__(self):
        """Initialization."""
        self.all_user = None
        self.all_iid = None
        self.tag2user = None
        self.tag2item = None

    def train(self,
              all_user: typing.Dict[int, typing.Dict[int, typing.Set[str]]]):
        """Training.

        Args:
            all_user, Dict[int, Dict[int, Set[str]]]. {uid: {iid: {tag}}}
        """
        # Initialization.
        self.all_user = all_user
        self.all_iid = set()

        # Build invert table.
        # Dict[str, Dict[int, int]]. {tag: {uid: count}}
        self.tag2user = collections.defaultdict(
            lambda: collections.defaultdict(int))
        # Dict[str, Dict[int, int]]. {tag: {iid: count}}
        self.tag2item = collections.defaultdict(
            lambda: collections.defaultdict(int))
        for uid, bought in self.all_user.items():
            for iid in bought:
                self.all_iid.add(iid)
                for tag in bought[iid]:
                    self.tag2user[tag][uid] += 1
                    self.tag2item[tag][iid] += 1



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
            iid: sum(self.tag2item[tag][iid] * self.tag2user[tag][uid]
                     for tag in self.tag2user)
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
    all_user = {1: {1: {'a'}, 2: {'b'}, 4: {'d'}}, 2: {1: {'a'}, 3: {'c'}},
                3: {2: {'c'}, 5: {'c'}}, 4: {3: {'b'}, 4: {'d', 'a'}, 5: {'a'}}}
    model = TagRecommendation()
    model.train(all_user)
    prediction, interest = model.predict(uid=1, N=2)
    print(prediction, interest)
    # Output:


if __name__ == '__main__':
    main()
