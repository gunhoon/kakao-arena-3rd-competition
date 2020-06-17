import argparse
import csv
import os
import subprocess
import pandas as pd


def main(count):
    out_fname = 'automate.csv'
    csv_header = ['dataset', 'Music nDCG', 'Tag nDCG', 'Score']

    with open(out_fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # 10개의 dataset을 만들어 성능 평가
    random_seed = [777, 710, 720, 730, 740, 750, 760, 770, 780, 790]
    if count < len(random_seed):
        random_seed = random_seed[0:count]

    for i in random_seed:
        print('---------->>> Start testing with random_seed: {}'.format(i))

        print('----->>> Run split_data.py ...')
        # python baseline/split_data.py run res/train.json --random_seed=777
        subprocess.run(['python', 'baseline/split_data.py', 'run',
                'res/train.json',
                '--random_seed={}'.format(i)], check=True)

        print('----->>> Run train.py ...')
        # python train.py --train_fname=arena_data/orig/train.json --question_fname=arena_data/questions/val.json
        subprocess.run(['python', 'train.py',
                '--train_fname=arena_data/orig/train.json',
                '--question_fname=arena_data/questions/val.json'], check=True)

        print('----->>> Run inference.py ...')
        # python inference.py --question_fname=arena_data/questions/val.json
        subprocess.run(['python', 'inference.py',
                '--question_fname=arena_data/questions/val.json'], check=True)

        print('----->>> Run evaluate.py ...')
        # python baseline/evaluate.py save_eval --gt_fname=arena_data/answers/val.json --rec_fname=arena_data/results/results.json
        subprocess.run(['python', 'baseline/evaluate.py', 'save_eval',
                '--gt_fname=arena_data/answers/val.json',
                '--rec_fname=arena_data/results/results.json',
                '--out_fname={}'.format(out_fname),
                '--dataset={}'.format(i)], check=True)

    print('---------->>> Calculate average ...')
    # automate.csv의 각 평균을 automate.csv에 write
    df = pd.read_csv(out_fname)

    m_avg = '{:.6}'.format(df['Music nDCG'].mean())
    t_avg = '{:.6}'.format(df['Tag nDCG'].mean())
    s_avg = '{:.6}'.format(df['Score'].mean())

    row = ['avg', m_avg, t_avg, s_avg]

    with open(out_fname, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print('---------->>> Finish testing ...')
    # automate.csv 전체를 console 로 출력
    with open(out_fname) as f:
        print(f.read())

    print('---------->>> Model Size ...')
    try:
        print(os.path.getsize('victoria.pickle'))
    except OSError:
        print('No victoria.pickle file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1, help='loop count')
    args = parser.parse_args()

    print(args)
    main(args.count)
