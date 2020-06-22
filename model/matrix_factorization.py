import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

from model.preference_data import PreferenceData


# Matrix Factorization using implicit library
class MatrixFactorization:
    def __init__(self, song_meta_json, song_topk, tag_topk):
        self._s_topk = song_topk
        self._t_topk = tag_topk

        self._s_model = AlternatingLeastSquares(factors=1300)
        self._t_model = AlternatingLeastSquares(factors=310)

        self._data = PreferenceData(song_meta_json)


    def fit(self, train, val):
        df = self._data.get_preference(train, val)

        s_len = self._data.get_song_length()
        t_len = self._data.get_tag_length()
        k_len = self._data.get_keyword_length()
        e_len = self._data.get_extension_length()
        val_len = self._data.get_val_length()

        # user x item csr_matrix
        user_item_csr = sparse.csr_matrix((df['preference'].astype(float), (df['user_id'], df['item_id'])))

        user_song_csr = user_item_csr[:, :s_len + t_len + k_len + e_len]
        user_tags_csr = user_item_csr

        print("Training song model...")
        self._s_model.fit(user_song_csr.T * 160)
        print("Training tag model...")
        self._t_model.fit(user_tags_csr.T * 40)

        # Configure song only model
        self._s_model.user_factors = self._s_model.user_factors[:val_len]
        self._s_model.item_factors = self._s_model.item_factors[:s_len]

        # Configure tag only model
        self._t_model.user_factors = self._t_model.user_factors[:val_len]
        self._t_model.item_factors = self._t_model.item_factors[s_len:s_len + t_len]

        # model size를 줄이기 위해 학습 끝난 후에, song_meta를 메모리에서 해제함.
        self._data.clear_song_meta()


    def predict(self, playlist):
        user_id = self._data.get_pid_to_uid(playlist['id'])

        s_best = []
        t_best = []
        if user_id != None:
            s_best = self._s_model.recommend(user_id, None, N=self._s_topk, filter_already_liked_items=False)
            t_best = self._t_model.recommend(user_id, None, N=self._t_topk, filter_already_liked_items=False)

        s_len = self._data.get_song_length()
        # s_best 는 (item_id, score)의 list, 이것을 (sid, score) 의 list로 변경
        s_pred = []
        t_pred = []
        for item_id, score in s_best:
            s_pred.append((self._data.get_iid_to_sid(item_id), score))
        for item_id, score in t_best:
            t_pred.append((self._data.get_iid_to_tag(item_id + s_len), score))

        return s_pred, t_pred
