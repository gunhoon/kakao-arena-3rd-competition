import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

from model.preference_data import PreferenceData


# Matrix Factorization using implicit library
class MatrixFactorization:
    def __init__(self, song_meta, song_topk, tag_topk):
        self._s_topk = song_topk
        self._t_topk = tag_topk

        self._s_best = None
        self._t_best = None

        self._s_data = PreferenceData(song_meta, song_type=True)
        self._t_data = PreferenceData(song_meta, song_type=False)


    def fit(self, train, val, extra_train):
        print('---> Starting the training for song...')
        # extra_train는 song에서만 사용함. tag에서 사용하면 성능 더 떨어짐.
        self._fit_for_song(train + extra_train, val)

        print('---> Starting the training for tag...')
        self._fit_for_tag(train, val)


    def _fit_for_song(self, train, val):
        df = self._s_data.get_preference(train, val)
        s_len = self._s_data.get_song_length()

        # user x item csr_matrix
        user_item_csr = sparse.csr_matrix((df['preference'].astype(float), (df['user_id'], df['item_id'])))

        s_model = AlternatingLeastSquares(factors=1600)
        s_model.fit(user_item_csr.T * 160)

        # Configure song only model
        s_model.item_factors = s_model.item_factors[:s_len]

        user_song_csr = user_item_csr[:, :s_len]
        self._s_best = s_model.recommend_all(user_song_csr, N=self._s_topk)


    def _fit_for_tag(self, train, val):
        df = self._t_data.get_preference(train, val)
        t_len = self._t_data.get_tag_length()

        # user x item csr_matrix
        user_item_csr = sparse.csr_matrix((df['preference'].astype(float), (df['user_id'], df['item_id'])))

        t_model = AlternatingLeastSquares(factors=1200)
        t_model.fit(user_item_csr.T * 65)

        # Configure tag only model
        t_model.item_factors = t_model.item_factors[:t_len]

        user_tags_csr = user_item_csr[:, :t_len]
        self._t_best = t_model.recommend_all(user_tags_csr, N=self._t_topk)


    def predict(self, playlist):
        s_pred = []
        t_pred = []

        user_id = self._s_data.get_pid_to_uid(playlist['id'])
        if user_id != None:
            for i, item_id in enumerate(self._s_best[user_id]):
                s_pred.append((self._s_data.get_iid_to_sid(int(item_id)), self._s_topk - i))

        user_id = self._t_data.get_pid_to_uid(playlist['id'])
        if user_id != None:
            for i, item_id in enumerate(self._t_best[user_id]):
                t_pred.append((self._t_data.get_iid_to_tag(int(item_id)), self._t_topk - i))

        return s_pred, t_pred
