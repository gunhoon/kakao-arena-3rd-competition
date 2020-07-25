from collections import Counter

from model.fallback import Fallback
from model.matrix_factorization import MatrixFactorization


class Victoria:
    def __init__(self, song_meta_json, song_topk=800, tag_topk=80):
        self._song_meta = {song['id'] : song for song in song_meta_json}

        self._s_topk = song_topk
        self._t_topk = tag_topk

        self._main_model = MatrixFactorization(self._song_meta, song_topk, tag_topk)
        self._fall_model = Fallback(song_topk, tag_topk)

        # 예외처리를 위해 발매일자가 잘못된 곡 저장
        self._issue_song = set()


    def fit(self, train, val, extra_train):
        self._find_issue_song(train + extra_train, val)

        self._main_model.fit(train, val, extra_train)
        self._fall_model.fit(train)


    def predict(self, playlist):
        s_pred, t_pred = self._main_model.predict(playlist)

        # MF된 결과에서 후처리를 하기 위해.
        s_pred = self._shift_position_song(s_pred, playlist)

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


    def _shift_position_song(self, s_pred, playlist):
        shifted_pred = []

        album_c = Counter()
        artist_c = Counter()

        for sid in playlist['songs']:
            album_c.update([self._song_meta[sid]['album_id']])
            artist_c.update(self._song_meta[sid]['artist_id_basket'])

        top_album = album_c.most_common(1)
        top_artist = artist_c.most_common(1)

        for sid, score in s_pred:
            shift_count = 0

            # playlist에 TOP 앨법의 곡이 5개 이상이면, 그 앨범의 모든 곡의 가중치를 높혀서
            # results.json 파일에서 앞쪽으로 배치시키기 위한 로직.
            if len(top_album) > 0 and top_album[0][0] == self._song_meta[sid]['album_id']:
                if top_album[0][1] > 4:
                    shift_count += self._s_topk

            # playlist에 특정 artist의 곡이 20개 이상이면, 그 artist의 모든 곡의 가중치를 높혀서
            # results.json 파일에서 앞쪽으로 배치시키기 위한 로직.
            # 또한, 10개 이상이면 100 칸 정도만 이동시킴.
            if len(top_artist) > 0 and top_artist[0][0] in self._song_meta[sid]['artist_id_basket']:
                if top_artist[0][1] > 19:
                    shift_count += self._s_topk
                elif top_artist[0][1] > 9:
                    shift_count += 100

            shifted_pred.append((sid, score + shift_count))

        return shifted_pred


    # 곡의 issue_date가 플레이리스트의 updt_date보다 늦은 곡 찾기
    def _find_issue_song(self, train, val):
        for p in train + val:
            updt_date = int(self._get_update_date(p))

            for sid in p['songs']:
                if int(self._song_meta[sid]['issue_date']) > updt_date:
                    self._issue_song.add(sid)


    # playlist의 updt_date와 song의 issue_date를 비교해서 재배치
    def _rearrange_song(self, s_pred, playlist):
        s_rec = []
        s_tmp = []
        updt_date = int(self._get_update_date(playlist))

        for sid, _ in s_pred:
            if int(self._song_meta[sid]['issue_date']) > updt_date and sid not in self._issue_song:
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
