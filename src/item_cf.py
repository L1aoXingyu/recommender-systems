"""item_cf.py

ItemCF model.
"""


import collections
import itertools
import math
import operator
import typing


class ItemCF():
    """Item CF model for recommendation.

    Attributes:
        item_similarity, Dict[int, Dict[int, float]]. {iid1: {iid2: similarity}}.
        all_user, Dict[int, Set[int]]. {uid: {iid}}
    """
    def __init__(self):
        self.item_similarity = None
        self.all_user = None

    def train(self, all_user: typing.Dict[int, typing.Set[int]], 
              penalty: bool = False):
        """Compute item similarity.
        
        Args:
            all_user, Dict[int, Set[int]]. {uid: {iid}}
            penalty, bool. Penalty for popular user.
        """
        # Save user data.
        self.all_user = all_user

        # Build item-user inverse table.
        all_item = collections.defaultdict(set)
        for uid, bought in all_user.items():
            for iid in bought:
                all_item[iid].add(uid)
                
        # Co-rated items betweens users.
        # Dict[int, Dict[int, int]]. 
        #     {iid1: {iid2: #(i \in f(u) \land i' \in f(u))}}
        self.item_similarity = collections.defaultdict(
            lambda: collections.defaultdict(float))
        for uid, bought in all_user.items():
            for iid1, iid2 in itertools.permutations(bought, 2):
                self.item_similarity[iid1][iid2] += (
                    1 if not penalty else 1 / math.log(1 + len(bought)))

        # Compute final similarity.
        for iid1 in self.item_similarity:
            for iid2 in self.item_similarity[iid1]:
                self.item_similarity[iid1][iid2] /= math.sqrt(
                    len(all_item[iid1]) * len(all_item[iid2]))
        
    def predict(self, uid: int, N: int, top_k: int) -> (
            typing.List[int], typing.List[float]):
        """Make prediction for the given user.

        Args:
            uid, int. User ID.
            N, int: Recommendation list length.
            top_k, int: Consider top-k similar items.
        
        Returns:
            prediction, List[int]. Predicted item ID.
            interest, List[float]. User interest to the given item ID.
            reason, List[int]. Because the given user bought.
        """
        # Dict[int, float]. {iid: interest}.
        interest_dict = collections.defaultdict(float)
        # Dict[int, int]. {iid: iid}.
        reason_dict = {}
        # Dict[int, float]. {iid: interest}.
        reason_value = collections.defaultdict(float)
        for iid in self.all_user[uid]:
            for iid2, similarity_value in sorted(
                    self.item_similarity[iid].items(), key=operator.itemgetter(1),
                    reverse=True)[:top_k]:
                if iid2 not in self.all_user[uid]:
                    interest_dict[iid2] += similarity_value
                    if similarity_value > reason_value[iid2]:
                        reason_value[iid2] = similarity_value
                        reason_dict[iid2] = iid

        # Make prediction.
        prediction = []
        interest = []
        reason = []
        for iid, interest_value in sorted(
                interest_dict.items(), key=operator.itemgetter(1),
                reverse=True)[:N]:
            prediction.append(iid)
            interest.append(interest_value)
            reason.append(reason_dict[iid])
        return prediction, interest, reason


def main():
    all_user = {1: {1, 2, 4}, 2: {1, 3}, 3: {2, 5}, 4: {3, 4, 5}}
    model = ItemCF()
    model.train(all_user)
    prediction, interest, reason = model.predict(uid=1, N=2, top_k=3)
    print(prediction, interest, reason)
    # Output: [3, 5] [1.0, 0.5]


if __name__ == '__main__':
    main()