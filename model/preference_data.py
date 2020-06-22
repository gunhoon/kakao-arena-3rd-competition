import pandas as pd
from collections import Counter

from model.song_meta import SongMeta
from model.tokenizer import Tokenizer


class PreferenceData:
    def __init__(self, song_meta_json):
        self._val_num = 0
        self._train_num = 0

        self._s_num = 0
        self._t_num = 0
        self._k_num = 0
        self._e_num = 0

        self._iid_to_sid = {}   # item_id to song_id
        self._iid_to_tag = {}   # item_id to tag
        self._pid_to_uid = {}   # plylst_id to user_id

        self._meta = SongMeta(song_meta_json)


    def clear_song_meta(self):
        self._meta = None


    def get_val_length(self):
        return self._val_num

    def get_train_length(self):
        return self._train_num

    def get_song_length(self):
        return self._s_num

    def get_tag_length(self):
        return self._t_num

    def get_keyword_length(self):
        return self._k_num

    def get_extension_length(self):
        return self._e_num


    def get_iid_to_sid(self, item_id):
        if item_id >= self._s_num:
            print('[ERROR] Invaild item_id of sid')
        return self._iid_to_sid.get(item_id)


    def get_iid_to_tag(self, item_id):
        # 0 base 로 변환
        item_id = item_id - self._s_num

        if item_id >= self._t_num:
            print('[ERROR] Invaild item_id of tag')
        return self._iid_to_tag.get(item_id)


    def get_pid_to_uid(self, plylst_id):
        return self._pid_to_uid.get(plylst_id)


    # Column(plylst_id, sid, tag, confidence, user_id, item_id)의 dataframe
    def get_preference(self, train, val):
        # 빈도가 적은 것은 feature에서 제거하기 위한 counter
        s_c = Counter()
        t_c = Counter()
 
        for p in train:
            s_c.update(p['songs'])
            t_c.update(p['tags'])
        for p in val:
            s_c.update(p['songs'])
            t_c.update(p['tags'])

        s_train, t_train, k_train, e_train, g_train = \
                self._preference_table(train, s_c, t_c)

        s_val, t_val, k_val, e_val, g_val = \
                self._preference_table(val, s_c, t_c)

        # song, tag, keyword, extenstion, genre of val
        s_val_df = pd.DataFrame(s_val, columns =['plylst_id', 'sid', 'preference'])
        t_val_df = pd.DataFrame(t_val, columns =['plylst_id', 'tag', 'preference'])
        k_val_df = pd.DataFrame(k_val, columns =['plylst_id', 'key', 'preference'])
        e_val_df = pd.DataFrame(e_val, columns =['plylst_id', 'ext', 'preference'])
        g_val_df = pd.DataFrame(g_val, columns =['plylst_id', 'gnr', 'preference'])
        # song, tag, keyword, extenstion, genre of train
        s_train_df = pd.DataFrame(s_train, columns =['plylst_id', 'sid', 'preference'])
        t_train_df = pd.DataFrame(t_train, columns =['plylst_id', 'tag', 'preference'])
        k_train_df = pd.DataFrame(k_train, columns =['plylst_id', 'key', 'preference'])
        e_train_df = pd.DataFrame(e_train, columns =['plylst_id', 'ext', 'preference'])
        g_train_df = pd.DataFrame(g_train, columns =['plylst_id', 'gnr', 'preference'])

        val_df = pd.concat([s_val_df, t_val_df, k_val_df, e_val_df, g_val_df])
        train_df = pd.concat([s_train_df, t_train_df, k_train_df, e_train_df, g_train_df])

        val_df['user_id'] = val_df['plylst_id'].astype('category').cat.codes
        train_df['user_id'] = train_df['plylst_id'].astype('category').cat.codes

        # plylst_id에서 user_id를 찾기 위한 dictionary 생성
        for _, r in val_df.iterrows():
            self._pid_to_uid.update({r['plylst_id']: r['user_id']})

        self._val_num = val_df['user_id'].nunique()
        self._train_num = train_df['user_id'].nunique()

        # train의 user_id는 val의 user_id 이후 번호 부터 부여한다.
        train_df['user_id'] = train_df['user_id'] + self._val_num

        df = pd.concat([val_df, train_df])

        s_df = df[ df['sid'].notna() ].reset_index()
        t_df = df[ df['tag'].notna() ].reset_index()
        k_df = df[ df['key'].notna() ].reset_index()
        e_df = df[ df['ext'].notna() ].reset_index()
        g_df = df[ df['gnr'].notna() ].reset_index()

        s_df['item_id'] = s_df['sid'].astype('category').cat.codes
        t_df['item_id'] = t_df['tag'].astype('category').cat.codes
        k_df['item_id'] = k_df['key'].astype('category').cat.codes
        e_df['item_id'] = e_df['ext'].astype('category').cat.codes
        g_df['item_id'] = g_df['gnr'].astype('category').cat.codes

        # item_id에서 sid를 찾기 위한 dictionary 생성
        for _, r in s_df.iterrows():
            self._iid_to_sid.update({r['item_id']: r['sid']})
        # item_id에서 tag를 찾기 위한 dictionary 생성
        for _, r in t_df.iterrows():
            self._iid_to_tag.update({r['item_id']: r['tag']})

        self._s_num = s_df['item_id'].nunique()
        self._t_num = t_df['item_id'].nunique()
        self._k_num = k_df['item_id'].nunique()
        self._e_num = e_df['item_id'].nunique()

        # tags의 item_id는 songs의 item_id 이후 번호 부터 부여한다.
        t_df['item_id'] = t_df['item_id'] + self._s_num
        k_df['item_id'] = k_df['item_id'] + self._s_num + self._t_num
        e_df['item_id'] = e_df['item_id'] + self._s_num + self._t_num + self._k_num
        g_df['item_id'] = g_df['item_id'] + self._s_num + self._t_num + self._k_num + self._e_num

        df = pd.concat([s_df, t_df, k_df, e_df, g_df])

        return df[['user_id', 'item_id', 'preference']].reset_index(drop=True)


    def _preference_table(self, playlists, s_counter, t_counter):
        s_table = []
        t_table = []
        k_table = []
        e_table = []
        g_table = []

        api = Tokenizer()

        for p in playlists:
            # songs
            for sid in p['songs']:
                if s_counter[sid] > 3:
                    s_table.append((p['id'], sid, 1))
                elif s_counter[sid] > 1:
                    e_table.append((p['id'], sid, 1))

            # tags
            for tag in p['tags']:
                if t_counter[tag] > 1:
                    t_table.append((p['id'], tag, 1))

            # keyword
            if len(p['plylst_title']) > 0:
                keyword = api.tokenize(p['plylst_title'])
                for k in keyword:
                    k_table.append((p['id'], k, 1))

            # detail genre
            gnr = self._meta.get_dtl_genre(p['songs'])
            for k, v in gnr:
                g_table.append((p['id'], k, v))

        return s_table, t_table, k_table, e_table, g_table
