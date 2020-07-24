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


    def fit(self, train, val):
        print("Training song model...")
        self._fit_for_song(train, val)

        print("Training tag model...")
        self._fit_for_tag(train, val)


    def _fit_for_song(self, train, val):
        df = self._s_data.get_preference(train, val)

        s_len = self._s_data.get_song_length()
        t_len = self._s_data.get_tag_length()
        k_len = self._s_data.get_keyword_length()
        e_len = self._s_data.get_extension_length()

        # user x item csr_matrix
        user_item_csr = sparse.csr_matrix((df['preference'].astype(float), (df['user_id'], df['item_id'])))
        user_song_csr = user_item_csr[:, :s_len + t_len + k_len + e_len]
        user_tags_csr = user_item_csr

        print("Training song model...")
        s_model = AlternatingLeastSquares(factors=1350)
        s_model.fit(user_song_csr.T * 160)

        # Configure song only model
        s_model.user_factors = s_model.user_factors
        s_model.item_factors = s_model.item_factors[:s_len]

        self._s_best = s_model.recommend_all(user_song_csr[:, :s_len], N=self._s_topk)


    def _fit_for_tag(self, train, val):
        df = self._t_data.get_preference(train, val)

        s_len = self._t_data.get_song_length()
        t_len = self._t_data.get_tag_length()
        k_len = self._t_data.get_keyword_length()
        e_len = self._t_data.get_extension_length()

        # user x item csr_matrix
        user_item_csr = sparse.csr_matrix((df['preference'].astype(float), (df['user_id'], df['item_id'])))
        user_song_csr = user_item_csr[:, :s_len + t_len + k_len + e_len]
        user_tags_csr = user_item_csr

        print("Training tag model...")
        t_model = AlternatingLeastSquares(factors=420)
        t_model.fit(user_tags_csr.T * 65)

        # Configure tag only model
        t_model.user_factors = t_model.user_factors
        t_model.item_factors = t_model.item_factors[s_len:s_len + t_len]

        self._t_best = t_model.recommend_all(user_tags_csr[:, s_len:s_len + t_len], N=self._t_topk)


    def predict(self, playlist):
        s_pred = []
        t_pred = []

        user_id = self._s_data.get_pid_to_uid(playlist['id'])
        s_len = self._s_data.get_song_length()

        if user_id != None:
            # s_best 는 (item_id, score)의 list, 이것을 (sid, score) 의 list로 변경
            for i, item_id in enumerate(self._s_best[user_id]):
                s_pred.append((self._s_data.get_iid_to_sid(int(item_id)), self._s_topk - i))


        user_id = self._t_data.get_pid_to_uid(playlist['id'])
        s_len = self._t_data.get_song_length()

        if user_id != None:
            for i, item_id in enumerate(self._t_best[user_id]):
                t_pred.append((self._t_data.get_iid_to_tag(int(item_id) + s_len), self._t_topk - i))

        return s_pred, t_pred
