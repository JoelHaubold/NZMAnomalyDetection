import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rrcf

plot_directory = Path("rrcf_plots")


def get_file_names(file_directory):
    file_names = os.listdir(file_directory)
    file_names = list(filter(lambda f_path: os.path.isdir(file_directory / f_path), file_names))
    return file_names


def plot_day(df_day, start_time):
    sdp_directory = plot_directory / "test_rrcf"
    if not os.path.exists(sdp_directory):
        os.makedirs(sdp_directory)

    plt.figure(1)
    plt.ylabel('Phases')
    p_counter = 1

    if not df_day.empty:
        df_day.Value.plot(figsize=(24, 6), linewidth=0.9, label="phase" + str(p_counter))
        df_day.AScore.plot(figsize=(24, 6), linewidth=0.5, color='grey', label="codisp", secondary_y=True)

    legend = plt.legend(fontsize='x-large', loc='lower left')
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    plot_path = sdp_directory / start_time

    plt.savefig(plot_path)
    plt.close(1)
    if df_day.AScore is not None:
        print(start_time + " --> " + str(max(df_day.AScore)))


def plot_anomaly_score(df, anomalie_scores):
    df['AScore'] = ((anomalie_scores.values - anomalie_scores.values.min()) / (
                anomalie_scores.values.max() - anomalie_scores.values.min()))
    day = pd.Timedelta('1d')
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    for start_time in pd.date_range(min_date, max_date, freq='d'):
        end_time = start_time + day
        df_day = df.loc[start_time:end_time]
        # ix_min_loc = df.index.get_loc(df_day.index[0])
        # ix_max_loc = df.index.get_loc(df_day.index[-1])
        # as_day = anomalie_scores.iloc[ix_min_loc:ix_max_loc+1]
        plot_day(df_day, str(start_time.date()))


def show_anomalies(df, anomalie_scores):
    df['AScore'] = anomalie_scores.values
    df99 = df[df.AScore > df.AScore.quantile(0.99)]
    print(df99)


def rrcf_calc(dfs):
    print(dfs)
    num_points = 0
    df_merged = dfs[0]
    for df in dfs:
        num_points += df.shape[0]
    print("num_points: " + str(num_points))
    num_trees = 10000
    tree_size = 1024
    shingle_size = 36

    points = rrcf.shingle(df_merged.Value, size=shingle_size)
    points = np.vstack([point for point in points])
    num_points = points.shape[0]
    sample_size_range = (num_points // tree_size, tree_size)
    forest = []
    while len(forest) < num_trees:
        print(len(forest))
        indices = np.random.choice(num_points, size=sample_size_range, replace=False)

        # trees = [rrcf.RCTree(df_merged.iloc[ix], index_labels=ix) for ix in indices]
        trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in indices]
        forest.extend(trees)

    avg_codisp = pd.Series(0.0, index=np.arange(num_points))
    n_owning_trees = np.zeros(num_points)
    for tree in forest:
        codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(n_owning_trees, codisp.index.values, 1)
    avg_codisp /= n_owning_trees
    avg_codisp.index = df_merged.Value.iloc[(shingle_size - 1):].index
    print(avg_codisp)
    plot_anomaly_score(df_merged, avg_codisp)


def prep_rrcf(pickle_directory, pickle_names):
    file_names = get_file_names(pickle_directory)
    dfs = []
    print(file_names)
    if pickle_names is not None:
        file_names = list(filter(lambda p: p in pickle_names, file_names))
    for f_name in file_names:
        f_path = pickle_directory / Path(f_name)
        df_phases = list(map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)).loc[:, ['Value']], ['1', '2', '3']))
        dfs += df_phases
    rrcf_calc([dfs[0]])

    # df = dfs[0]
    # plot_anomaly_score(df, pd.read_csv("0888_p1_as_10k",index_col=0, squeeze=True, header=None))


def rrcf_calc_codisp(df, num_trees, tree_size, shingle_size):
    print(df)
    num_points = df.shape[0]
    print("num_points: " + str(num_points))

    points = rrcf.shingle(df.Value, size=shingle_size)
    points = np.vstack([point for point in points])
    num_points = points.shape[0]
    sample_size_range = (num_points // tree_size, tree_size)
    forest = []
    while len(forest) < num_trees:
        print(len(forest))
        indices = np.random.choice(num_points, size=sample_size_range, replace=False)

        # trees = [rrcf.RCTree(df_merged.iloc[ix], index_labels=ix) for ix in indices]
        trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in indices]
        forest.extend(trees)

    avg_codisp = pd.Series(0.0, index=np.arange(num_points))
    n_owning_trees = np.zeros(num_points)
    for tree in forest:
        codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(n_owning_trees, codisp.index.values, 1)
    avg_codisp /= n_owning_trees
    avg_codisp.index = df.Value.iloc[(shingle_size - 1):].index
    print(avg_codisp)
    avg_codisp.to_csv()


def main():
    pickle_directory = Path("pickles")
    test_pickle = ['NW000000000000000000000NBSNST0888']
    df = prep_rrcf(pickle_directory, test_pickle)


if __name__ == '__main__':
    main()
