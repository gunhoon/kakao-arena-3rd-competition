from collections import Counter


class SongMeta:
    def __init__(self, song_meta_json):
        self._song_meta = {song['id'] : song for song in song_meta_json}


    def get_genre(self, songs):
        c = Counter()

        for sid in songs:
            value = self._song_meta[sid]['song_gn_gnr_basket']
            c.update(value)

        return c.items()


    def get_dtl_genre(self, songs):
        c = Counter()

        for sid in songs:
            value = self._song_meta[sid]['song_gn_dtl_gnr_basket']
            c.update(value)

        return c.items()


    def get_album(self, songs):
        c = Counter()

        for sid in songs:
            value = self._song_meta[sid]['album_id']
            c.update([value])

        return c.items()


    def get_artist(self, songs):
        c = Counter()

        for sid in songs:
            value = self._song_meta[sid]['artist_id_basket']
            c.update(value)

        return c.items()
