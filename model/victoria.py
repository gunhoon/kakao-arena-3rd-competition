from collections import Counter

from model.fallback import Fallback
from model.matrix_factorization import MatrixFactorization


class Victoria:
    def __init__(self, song_meta_json, song_topk=200, tag_topk=20):
        self._main_model = MatrixFactorization(song_meta_json, song_topk, tag_topk)
        self._fall_model = Fallback(song_topk, tag_topk)


    def fit(self, train, val):
        self._main_model.fit(train, val)
        self._fall_model.fit(train)


    def predict(self, playlist):
        s_pred, t_pred = self._main_model.predict(playlist)

        # fallback model
        if len(s_pred) == 0:
            s_pred, _ = self._fall_model.predict(playlist)
        if len(t_pred) == 0:
            _, t_pred = self._fall_model.predict(playlist)

        s_rec = [k for k, v in s_pred]
        t_rec = [k for k, v in t_pred]

        return s_rec, t_rec
