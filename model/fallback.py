from collections import Counter


class Fallback:
    def __init__(self, song_topk, tag_topk):
        self._s_topk = song_topk
        self._t_topk = tag_topk
        self._s_pred = None
        self._t_pred = None


    def fit(self, train):
        s_c = Counter()
        t_c = Counter()

        for p in train:
            s_c.update(p['songs'])
            t_c.update(p['tags'])

        self._s_pred = s_c.most_common(self._s_topk)
        self._t_pred = t_c.most_common(self._t_topk)


    def predict(self, playlist):
        # list of tuple : [(sid, counter), (sid, counter), ...]
        return self._s_pred, self._t_pred
