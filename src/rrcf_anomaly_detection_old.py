import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rrcf

rrcf_directory = Path("../../rrcf")
pickle_directory = Path("../../pickles")
num_trees = 5000
tree_size = 2000
shingle_size = 2
stream = False
codisp_filename = f"Shingle{shingle_size}_TreeN{num_trees}_TreeS{tree_size}"


def get_file_names(file_directory):
    file_names = os.listdir(file_directory)
    file_names = list(filter(lambda f_path: os.path.isdir(file_directory / f_path), file_names))
    return file_names


def plot_day(df_day, start_time, station_name):
    sdp_directory = rrcf_directory / station_name
    if not os.path.exists(sdp_directory):
        os.makedirs(sdp_directory)

    plt.figure()
    plt.ylabel('Phases')
    p_counter = 1

    if not df_day.empty:
        ax1 = df_day.Value.plot(figsize=(24, 6), linewidth=0.9, label="phase" + str(p_counter))
        ax2 = df_day.AScore.plot(figsize=(24, 6), linewidth=0.5, color='grey', label="codisp", secondary_y=True)
        ax2.set_ylim([0, 1])
    legend = plt.legend(fontsize='x-large', loc='lower left')
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    plot_path = sdp_directory / start_time

    plt.savefig(plot_path)
    plt.close()
    if df_day.AScore is not None:
        print(start_time + " --> " + str(max(df_day.AScore)))


def plot_anomaly_score(df, anomalie_scores, station_name):
    # df.index = anomalie_scores.index
    as_norm = ((anomalie_scores.values - anomalie_scores.values.min()) / (
            anomalie_scores.values.max() - anomalie_scores.values.min()))
    if df.shape[0] > as_norm.__len__():
        print(f"fixed length from {df.shape[0]} to {as_norm.__len__}")
        df = df.head(as_norm.__len__())
    df['AScore'] = as_norm
    print(df)
    day = pd.Timedelta('1d')
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    for start_time in pd.date_range(min_date, max_date, freq='d'):
        end_time = start_time + day
        df_day = df.loc[start_time:end_time]
        # ix_min_loc = df.index.get_loc(df_day.index[0])
        # ix_max_loc = df.index.get_loc(df_day.index[-1])
        # as_day = anomalie_scores.iloc[ix_min_loc:ix_max_loc+1]
        plot_day(df_day, str(start_time.date()),station_name)


def analyse_station_phase(df_p, station_name):
    sdp_directory = rrcf_directory / station_name
    if not os.path.exists(sdp_directory):
        os.makedirs(sdp_directory)

    # Prepare points
    num_points = df_p.shape[0]
    print(num_points)
    if shingle_size is not None:
        points = rrcf.shingle(df_p.Value, size=shingle_size)
        points = np.vstack([point for point in points])
        num_points = points.shape[0]
    sample_size_range = (num_points // tree_size, tree_size)
    forest = []

    # Construct forest
    while len(forest) < num_trees:
        print(len(forest))
        indices = np.random.choice(num_points, size=sample_size_range, replace=False)

        if shingle_size is not None:
            trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in indices]
        else:
            trees = [rrcf.RCTree(df_p.iloc[ix], index_labels=ix) for ix in indices]
        forest.extend(trees)

    # Calculate Codisp
    avg_codisp = pd.Series(0.0, index=np.arange(num_points))
    n_owning_trees = np.zeros(num_points)
    for tree in forest:
        tree.
        codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(n_owning_trees, codisp.index.values, 1)
    avg_codisp /= n_owning_trees

    # Save Codisp
    avg_codisp.to_json(sdp_directory / codisp_filename)


def main():
    test_pickle = 'NW000000000000000000000NBSNST0888'
    plot_dir = test_pickle + "p1"
    # file_names = get_file_names(pickle_directory)
    # file_names = list(filter(lambda p: p in test_pickle, file_names))
    f_path = pickle_directory / test_pickle
    df_phases = list(
        map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)).loc[:, ['Value']], ['1', '2', '3']))
    df_p = df_phases[0]
    analyse_station_phase(df_p, plot_dir)
    avg_codisp = pd.read_json(Path(f"../../rrcf/{plot_dir}/{codisp_filename}"), typ='series')
    plot_anomaly_score(df_p, avg_codisp, plot_dir)


if __name__ == '__main__':
    main()
