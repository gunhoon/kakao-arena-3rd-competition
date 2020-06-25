import argparse
import os
import pickle

from baseline.arena_util import load_json
from model.victoria import Victoria


def main(song_meta_fname, train_fname, question_fname):
    print('Loading song meta...')
    song_meta_json = load_json(song_meta_fname)

    print('Loading train file...')
    train = load_json(train_fname)

    print('Loading question file...')
    question = load_json(question_fname)

    print('Training a model...')
    model = Victoria(song_meta_json)
    model.fit(train, question)

    print('Saving the model...')
    with open('victoria.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Printing the model size...')
    try:
        print(os.path.getsize('victoria.pkl'))
    except OSError:
        print('No victoria.pkl file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--song_meta_fname', default='res/song_meta.json')
    parser.add_argument('--train_fname', default='res/train.json')
    parser.add_argument('--question_fname', default='res/test.json')
    args = parser.parse_args()

    print(args)
    main(args.song_meta_fname, args.train_fname, args.question_fname)
