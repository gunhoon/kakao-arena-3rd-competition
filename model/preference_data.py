import pandas as pd
from collections import Counter
from tqdm import tqdm

from model.tokenizer import Tokenizer


class PreferenceData:
    def __init__(self, song_meta, song_type=True):
        self._song_meta = song_meta
        # song과 tag를 따로 분리하기 위해(성능 향상에 차이가 있어서)
        self._song_type = song_type

        self._s_num = 0
        self._t_num = 0
        self._k_num = 0
        self._e_num = 0

        self._iid_to_sid = {}   # item_id to song_id
        self._iid_to_tag = {}   # item_id to tag
        self._pid_to_uid = {}   # plylst_id to user_id


    def get_song_length(self):
        return self._s_num

    def get_tag_length(self):
        return self._t_num

    def get_keyword_length(self):
        return self._k_num

    def get_extension_length(self):
        return self._e_num


    def get_iid_to_sid(self, item_id):
        # 0 base 로 변환
        if not self._song_type:
            item_id = item_id - self._t_num

        if item_id >= self._s_num:
            print('[ERROR] Invaild item_id of sid')
        return self._iid_to_sid.get(item_id)


    def get_iid_to_tag(self, item_id):
        # 0 base 로 변환
        if self._song_type:
            item_id = item_id - self._s_num

        if item_id >= self._t_num:
            print('[ERROR] Invaild item_id of tag')
        return self._iid_to_tag.get(item_id)


    def get_pid_to_uid(self, plylst_id):
        return self._pid_to_uid.get(plylst_id)


    def get_preference(self, train, val):
        if self._song_type:
            df = self._get_preference_for_song(train, val)
        else:
            df = self._get_preference_for_tag(train, val)

        return df


    def _get_preference_for_song(self, train, val):
        s_table, t_table, k_table = \
                self._preference_table_for_song(train + val)

        # song, tag, keyword
        s_df = pd.DataFrame(s_table, columns =['plylst_id', 'sid', 'preference'])
        t_df = pd.DataFrame(t_table, columns =['plylst_id', 'tag', 'preference'])
        k_df = pd.DataFrame(k_table, columns =['plylst_id', 'key', 'preference'])

        s_df['item_id'] = s_df['sid'].astype('category').cat.codes
        t_df['item_id'] = t_df['tag'].astype('category').cat.codes
        k_df['item_id'] = k_df['key'].astype('category').cat.codes
 
        # item_id에서 sid를 찾기 위한 dictionary 생성
        new_df = s_df[['item_id', 'sid']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._iid_to_sid.update({r['item_id']: r['sid']})
        # item_id에서 tag를 찾기 위한 dictionary 생성
        new_df = t_df[['item_id', 'tag']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._iid_to_tag.update({r['item_id']: r['tag']})

        self._s_num = s_df['item_id'].nunique()
        self._t_num = t_df['item_id'].nunique()
        self._k_num = k_df['item_id'].nunique()

        # tags의 item_id는 songs의 item_id 이후 번호 부터 부여한다(keyword도 마찬가지).
        t_df['item_id'] = t_df['item_id'] + self._s_num
        k_df['item_id'] = k_df['item_id'] + self._s_num + self._t_num

        df = pd.concat([s_df, t_df, k_df])

        df['user_id'] = df['plylst_id'].astype('category').cat.codes

        # plylst_id를 user_id를 찾기 위한 dictionary 생성
        new_df = df[['plylst_id', 'user_id']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._pid_to_uid.update({r['plylst_id']: r['user_id']})

        return df[['user_id', 'item_id', 'preference']].reset_index(drop=True)


    def _get_preference_for_tag(self, train, val):
        s_table, t_table, k_table, e_table, g_table = \
                self._preference_table_for_tag(train + val)

        # tag, song, keyword, extenstion, genre
        t_df = pd.DataFrame(t_table, columns =['plylst_id', 'tag', 'preference'])
        s_df = pd.DataFrame(s_table, columns =['plylst_id', 'sid', 'preference'])
        k_df = pd.DataFrame(k_table, columns =['plylst_id', 'key', 'preference'])
        e_df = pd.DataFrame(e_table, columns =['plylst_id', 'ext', 'preference'])
        g_df = pd.DataFrame(g_table, columns =['plylst_id', 'gnr', 'preference'])

        t_df['item_id'] = t_df['tag'].astype('category').cat.codes
        s_df['item_id'] = s_df['sid'].astype('category').cat.codes
        k_df['item_id'] = k_df['key'].astype('category').cat.codes
        e_df['item_id'] = e_df['ext'].astype('category').cat.codes
        g_df['item_id'] = g_df['gnr'].astype('category').cat.codes

        # item_id에서 sid를 찾기 위한 dictionary 생성
        new_df = s_df[['item_id', 'sid']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._iid_to_sid.update({r['item_id']: r['sid']})
        # item_id에서 tag를 찾기 위한 dictionary 생성
        new_df = t_df[['item_id', 'tag']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._iid_to_tag.update({r['item_id']: r['tag']})

        self._t_num = t_df['item_id'].nunique()
        self._s_num = s_df['item_id'].nunique()
        self._k_num = k_df['item_id'].nunique()
        self._e_num = e_df['item_id'].nunique()

        # song의 item_id는 tag의 item_id 이후 번호 부터 부여한다.
        s_df['item_id'] = s_df['item_id'] + self._t_num
        k_df['item_id'] = k_df['item_id'] + self._t_num + self._s_num
        e_df['item_id'] = e_df['item_id'] + self._t_num + self._s_num + self._k_num
        g_df['item_id'] = g_df['item_id'] + self._t_num + self._s_num + self._k_num + self._e_num

        df = pd.concat([t_df, s_df, k_df, e_df, g_df])

        df['user_id'] = df['plylst_id'].astype('category').cat.codes

        # plylst_id를 user_id를 찾기 위한 dictionary 생성
        new_df = df[['plylst_id', 'user_id']].drop_duplicates().reset_index(drop=True)
        for _, r in new_df.iterrows():
            self._pid_to_uid.update({r['plylst_id']: r['user_id']})

        return df[['user_id', 'item_id', 'preference']].reset_index(drop=True)


    def _preference_table_for_song(self, playlists):
        s_table = []
        t_table = []
        k_table = []

        # 빈도가 적은 것은 feature에서 제거하기 위한 counter
        s_c = Counter()
        t_c = Counter()
 
        for p in playlists:
            s_c.update(p['songs'])
            t_c.update(p['tags'])

        api = Tokenizer()

        for p in tqdm(playlists):
            # songs
            for sid in p['songs']:
                if s_c[sid] > 1:
                    s_table.append((p['id'], sid, 1.0))

            # tags
            for tag in p['tags']:
                if t_c[tag] > 1:
                    t_table.append((p['id'], tag, 0.5))

            # keyword
            if len(p['plylst_title']) > 0:
                keyword = api.tokenize(p['plylst_title'])
                for k in keyword:
                    k_table.append((p['id'], k, 0.5))

        return s_table, t_table, k_table


    def _preference_table_for_tag(self, playlists):
        s_table = []
        t_table = []
        k_table = []
        e_table = []
        g_table = []

        # 빈도가 적은 것은 feature에서 제거하기 위한 counter
        s_c = Counter()
        t_c = Counter()
 
        for p in playlists:
            s_c.update(p['songs'])
            t_c.update(p['tags'])

        api = Tokenizer()

        for p in tqdm(playlists):
            # songs
            for sid in p['songs']:
                if s_c[sid] > 3:
                    s_table.append((p['id'], sid, 1))
                # 3 이하는 참조는 하는데, 추천은 하지 않음.
                elif s_c[sid] > 1:
                    e_table.append((p['id'], sid, 1))

            # tags
            for tag in p['tags']:
                if t_c[tag] > 1:
                    t_table.append((p['id'], tag, 1))

            # keyword
            if len(p['plylst_title']) > 0:
                keyword = api.tokenize(p['plylst_title'])
                for k in keyword:
                    k_table.append((p['id'], k, 1))

            # detail genre
            gnr = self._get_dtl_genre(p['songs'])
            for k, v in gnr:
                g_table.append((p['id'], k, v))

        return s_table, t_table, k_table, e_table, g_table


    def _get_dtl_genre(self, songs):
        c = Counter()

        for sid in songs:
            value = self._song_meta[sid]['song_gn_dtl_gnr_basket']
            c.update(value)

        return c.items()
