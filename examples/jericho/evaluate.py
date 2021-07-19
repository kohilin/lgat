import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import seaborn as sns

from tensorboard.backend.event_processing import event_accumulator


MAX_SCORES = {
    'balances': 51, 'deephome': 300, 'detective': 360, 'enchanter': 400,
    'gold': 100, 'inhumane': 90, 'jewel': 90, 'library': 30, 'ludicorp': 150,
    'omniquest': 50, 'reverb': 50, 'snacktime': 50, 'spellbrkr': 600,
    'spirit': 250, 'temple': 35, 'zork1': 350, 'zork3': 7, 'ztuu': 100
}

MODELS = ['L-GAT', 'TDQN+']


def load_score_avg(f):
    ea = event_accumulator.EventAccumulator(f)
    ea.Reload()
    values = ea.Scalars('score/avg')
    v = np.stack([np.asarray([scalar.step, scalar.value]) for scalar in values])
    return v


def ranged_mean_score(x, end=1000, r=10):
    end = min(end, x.shape[0])
    begin = max(end - r, 0)
    return x[begin:end, [1]].mean()


def moving_average(x, w=10):
    return np.convolve(x, np.ones(w), 'valid') / w


def compute_scores(args):
    files = glob.glob(args.tb_event_dir + '/**/*tfevents*', recursive=True)

    def __create_dict():
        d = dict()
        for g in args.target_games:
            d[g] = {}
            for m in MODELS:
                d[g][m] = []
        return d

    tmp = []
    for f in files:
        for g in args.target_games:
            if g in f:
                tmp.append(f)
    files = tmp

    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(load_score_avg, files)

    episode_scores = __create_dict()
    for f, r in zip(files, results):
        for g in args.target_games:
            if g in f:
                if 'tdqn' in f:
                    episode_scores[g][MODELS[1]].append(r)
                else:
                    episode_scores[g][MODELS[0]].append(r)

    lines = []
    for g, v in episode_scores.items():
        for m, ess in v.items():
            for es in ess:
                score = ranged_mean_score(es)
                norm_score = score / MAX_SCORES[g]
                lines.append((g, m, score, norm_score))

    df = pd.DataFrame(lines, columns=['game', 'model', 'score', 'norm_score'])
    results = df.groupby(['game', 'model']).mean().reset_index('model')[['model', 'score']]
    results['file_nums'] = df.groupby(['game', 'model']).count()['score'].tolist()
    results_norm_score = df.groupby('game').mean()[['norm_score']]
    results_norm_score_across_games = df.groupby('model').mean()[['norm_score']]

    print(">>> scores")
    print(results)
    print()

    print(">>> norm scores")
    print(results_norm_score)
    print()

    print(">>> norm scores across games")
    print(results_norm_score_across_games)
    print()

    return episode_scores


def plot(episode_scores, args):
    if args.plot_save_dir is None:
        save_path = os.path.join(args.tb_event_dir, 'plot')
    else:
        save_path = args.plot_save_dir

    os.makedirs(save_path, exist_ok=True)
    for g, v in episode_scores.items():
        lines = []
        for m, ess in v.items():
            for es in ess:
                mavg = moving_average(es[:, 1])
                for idx, s in enumerate(mavg):
                    lines.append((g, m, idx, s))

        df = pd.DataFrame(lines, columns=['Game', 'Model', 'Episode', 'Score'])

        plt.figure()
        ax = sns.lineplot(data=df,
                          x='Episode', y='Score',
                          hue='Model', ci='sd', hue_order=MODELS)
        ax.set_title(g)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')
        ax.get_figure().savefig(f'{os.path.join(save_path, g)}.png')


def main(args):
    episode_scores = compute_scores(args)
    plot(episode_scores, args)


if __name__ == '__main__':
    from argparse import ArgumentParser

    print(">>> [Warning] This script assumes results with epoch=1000 and env_batch_size=10.")

    parser = ArgumentParser(add_help=True)

    parser.add_argument('--tb_event_dir')

    parser.add_argument('--target_games',
                        nargs='+',
                        default=['balances', 'deephome', 'detective',
                                 'enchanter', 'gold', 'inhumane', 'jewel',
                                 'library', 'ludicorp', 'omniquest', 'reverb',
                                 'snacktime', 'spellbrkr', 'spirit', 'temple',
                                 'zork1', 'zork3', 'ztuu'])

    parser.add_argument('--plot_save_dir',
                        default=None)

    args = parser.parse_args()

    main(args)
