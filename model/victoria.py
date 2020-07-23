from collections import Counter

from model.fallback import Fallback
from model.matrix_factorization import MatrixFactorization


class Victoria:
    def __init__(self, song_meta_json, song_topk=500, tag_topk=20):
        self._main_model = MatrixFactorization(song_meta_json, song_topk, tag_topk)
        self._fall_model = Fallback(song_topk, tag_topk)

        self._issue_song = set() # 예외처리를 위해 발매일자가 잘못된 곡 저장
        self._issue_date = {song['id'] : song['issue_date'] for song in song_meta_json}

        self._album_id = {song['id'] : song['album_id'] for song in song_meta_json}
        self._artist_id = {song['id'] : song['artist_id_basket'] for song in song_meta_json}


    def fit(self, train, val):
        self._find_issue_song(train, val)

        self._main_model.fit(train, val)
        self._fall_model.fit(train)


    def predict(self, playlist):
        s_pred, t_pred = self._main_model.predict(playlist)

        s_pred = self._set_song_weight(s_pred, playlist)

        s_pred.sort(key=lambda tup: tup[1], reverse=True)
        t_pred.sort(key=lambda tup: tup[1], reverse=True)

        # fallback model
        if len(s_pred) == 0:
            s_pred, _ = self._fall_model.predict(playlist)
        if len(t_pred) == 0:
            _, t_pred = self._fall_model.predict(playlist)

        s_rec = self._rearrange_song(s_pred, playlist)
        t_rec = [k for k, v in t_pred]

        return s_rec, t_rec

    # 가중치 함수
    def _set_song_weight(self, s_pred, playlist):
        w_pred = []

        album_c = Counter()
        artist_c = Counter()

        for sid in playlist['songs']:
            album_c.update([self._album_id[sid]])
            artist_c.update(self._artist_id[sid])

        album_dict = dict(album_c)
        artist_dict = dict(artist_c)

        for sid, score in s_pred:
            album_weight = album_dict.get(self._album_id[sid])
            if album_weight == None:
                album_weight = 0

            artist_weight = 0
            for artist_id in self._artist_id[sid]:
                weight = artist_dict.get(artist_id)
                if weight == None:
                    weight = 0
                if weight > artist_weight:
                    artist_weight = weight

            new_score = score + (album_weight + artist_weight) * 0.1
            w_pred.append((sid, new_score))

        return w_pred


    # 곡의 issue_date가 플레이리스트의 updt_date보다 늦은 곡 찾기
    def _find_issue_song(self, train, val):
        for p in train + val:
            updt_date = int(self._get_update_date(p))

            for sid in p['songs']:
                if int(self._issue_date[sid]) > updt_date:
                    self._issue_song.add(sid)


    # playlist의 updt_date와 song의 issue_date를 비교해서 재배치
    def _rearrange_song(self, s_pred, playlist):
        s_rec = []
        s_tmp = []
        updt_date = int(self._get_update_date(playlist))

        for sid, _ in s_pred:
            if int(self._issue_date[sid]) > updt_date and sid not in self._issue_song:
                s_tmp.append(sid)
            else:
                s_rec.append(sid)

        s_rec.extend(s_tmp)
        return s_rec


    # playlist의 updt_date
    def _get_update_date(self, playlist):
        updt_date = playlist['updt_date'][0:4] + \
                    playlist['updt_date'][5:7] + \
                    playlist['updt_date'][8:10]

        return updt_date
