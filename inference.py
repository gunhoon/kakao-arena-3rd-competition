import argparse
import pickle

from tqdm import tqdm

from baseline.arena_util import load_json
from baseline.arena_util import write_json
from baseline.arena_util import remove_seen
from model.victoria import Victoria


def main(question_fname):
    print('Loading question file...')
    question = load_json(question_fname)

    print('Loading model...')
    with open('victoria.pickle', 'rb') as f:
        model = pickle.load(f)

    print('Creating answers...')
    answers = []

    for q in tqdm(question):
        s_rec, t_rec = model.predict(q)

        answers.append({
            'id': q['id'],
            'songs': remove_seen(q['songs'], s_rec)[:100],
            'tags': remove_seen(q['tags'], t_rec)[:10],
        })

    print('Writing answers...')
    write_json(answers, 'results/results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_fname', default='res/test.json')
    args = parser.parse_args()

    print(args)
    main(args.question_fname)
