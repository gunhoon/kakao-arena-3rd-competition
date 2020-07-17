import re

from khaiii import KhaiiiApi


class Tokenizer:
    def __init__(self):
        self._api = KhaiiiApi()
        # 불용어 정의
        self._stopwords = ['말', '곡', '때', '음악', '노래',
                'a', 'an', 'the', 'in', 'on', 'at', 'by', 'of']

        # 대체어
        self._alternative = [
            ('k-pop', 'kpop'),
            ('k팝', 'kpop'),
            ('j-pop', 'jpop'),
            ('r&b', 'rnb'),
            ('알앤비', 'rnb'),
            ('락', 'rock'),
            ('재즈', 'jazz'),
            ('째즈', 'jazz'),
            ('힙합', 'hiphop'),
            ('hip-hop', 'hiphop'),
            ('hip-hap', 'hiphop'),
            ('클래식', 'classic'),
            ('발라드', ' 발라드 '),
            ('라붐', 'laboum'),
            ('뉴에이지', 'newage'),
        ]


    def tokenize(self, sentence):
        clean_sentence = sentence.lower()

        # 대체어로 대체
        for words in self._alternative:
            clean_sentence = re.sub(words[0], words[1], clean_sentence)

        # 영어는 소문자로, 그리고 숫자/영어/한글을 제외한 특수문자 제거(ㅋ 포함)
        clean_sentence = re.sub('[^0-9a-z가-힣]', ' ', clean_sentence)

        morphs = []
        try:
            for word in self._api.analyze(clean_sentence):
                morphs.extend(self._word_tokenize(word))
        except:
            morphs.clear()
            print('[WARNING] Khaiii can not tokenize...({})'.format(sentence))

        # 불용어 제거
        keyword = {lex for lex, _ in morphs if not lex in self._stopwords}

        return list(keyword)


    def _word_tokenize(self, word):
        morphs = []

        prev_lex = ''
        prev_tag = ''

        for morph in word.morphs:
            # 복합명사는 복합명사 그대로 저장
            if morph.tag == 'NNG' and prev_tag == 'NNG':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'NNG' and prev_tag == 'NNP':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'NNP' and prev_tag == 'NNG':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'NNP' and prev_tag == 'NNP':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))

            elif morph.tag == 'NNG' and prev_tag == 'XR':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'NNP' and prev_tag == 'XR':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'XR' and prev_tag == 'NNG':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))
            elif morph.tag == 'XR' and prev_tag == 'NNP':
                morphs.append((morphs.pop()[0] + morph.lex, morph.tag))

            elif morph.tag == 'NNG' and prev_tag == 'IC':
                morphs.append((prev_lex + morph.lex, morph.tag))
            elif morph.tag == 'NNP' and prev_tag == 'IC':
                morphs.append((prev_lex + morph.lex, morph.tag))

            # 일반명사
            elif morph.tag == 'NNG':
                morphs.append((morph.lex, morph.tag))
            # 고유명사
            elif morph.tag == 'NNP':
                morphs.append((morph.lex, morph.tag))
            # 외국어
            elif morph.tag == 'SL':
                morphs.append((morph.lex, morph.tag))
            # 어근
            elif morph.tag == 'XR':
                morphs.append((morph.lex, morph.tag))

            # 동사 : 2자리 이상만
            elif morph.tag == 'VV' and len(morph.lex) > 1:
                morphs.append((morph.lex, morph.tag))
            # 형용사 : 2자리 이상만
            elif morph.tag == 'VA' and len(morph.lex) > 1:
                morphs.append((morph.lex, morph.tag))

            # 숫자 : 2자리 이상만
            elif morph.tag == 'SN' and len(morph.lex) > 1:
                morphs.append((morph.lex, morph.tag))
            # 숫자 + 의존명사 (예, 2000년대)
            elif morph.tag == 'NNB' and prev_tag == 'SN':
                morphs.append((prev_lex + morph.lex, morph.tag))

            prev_lex = morph.lex
            prev_tag = morph.tag

        return morphs
