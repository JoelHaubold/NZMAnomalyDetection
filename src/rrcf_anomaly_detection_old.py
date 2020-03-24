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
shingle_size = None
streaming = True
codisp_filename = f"Shingle{shingle_size}_TreeN{num_trees}_TreeS{tree_size}Streaming{streaming}"


def get_file_names(file_directory):
    file_names = os.listdir(file_directory)
    file_names = list(filter(lambda f_path: os.path.isdir(file_directory / f_path), file_names))
    return file_names


def plot_phase_dif(df, non_streaming_score, streaming_score, station_name):
    stream_norm = ((streaming_score.values - streaming_score.values.min()) / (
            streaming_score.values.max() - streaming_score.values.min()))
    non_stream_norm = ((non_streaming_score.values - non_streaming_score.values.min()) / (
            non_streaming_score.values.max() - non_streaming_score.values.min()))
    df['ns_score'] = non_stream_norm
    df['s_score'] = stream_norm
    day = pd.Timedelta('1d')
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    for start_time in pd.date_range(min_date, max_date, freq='d'):
        end_time = start_time + day
        df_day = df.loc[start_time:end_time]
        plt.figure()
        plt.ylabel('Phases')
        p_counter = 1

        if not df_day.empty:
            df_day.p1.plot(figsize=(24, 6), linewidth=0.9, label="phase" + str(1))
            df_day.p2.plot(figsize=(24, 6), linewidth=0.9, label="phase" + str(2))
            df_day.p3.plot(figsize=(24, 6), linewidth=0.9, label="phase" + str(3))
            ax2 = df_day.s_score.plot(figsize=(24, 6), linewidth=0.5, color='grey', label="s_codisp", secondary_y=True)
            ax2.set_ylim([0, 1])
            df_day.ns_score.plot(figsize=(24, 6), linewidth=0.5, color='brown', label="ns_codisp", secondary_y=True)
        legend = plt.legend(fontsize='x-large', loc='lower left')
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        sdp_directory = rrcf_directory / station_name / codisp_filename
        plot_path = sdp_directory / str(start_time)

        plt.savefig(plot_path)
        plt.close()


def plot_day(df_day, start_time, station_name):
    sdp_directory = rrcf_directory / station_name / codisp_filename
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
        plot_day(df_day, str(start_time.date()), station_name)


def analyse_station_phase(df_p, station_name):
    sdp_directory = rrcf_directory / station_name / codisp_filename
    if not os.path.exists(sdp_directory):
        os.makedirs(sdp_directory)

    # Prepare points
    num_points = df_p.shape[0]
    print(num_points)
    if shingle_size is not None:
        points = rrcf.shingle(df_p, size=shingle_size)
        points = np.vstack([point for point in points])
        num_points = points.shape[0]
    sample_size_range = (num_points // tree_size, tree_size)
    forest = []

    # Construct forest
    if streaming:
        while len(forest) < num_trees:
            print(len(forest))
            indices = np.random.choice(num_points, size=sample_size_range, replace=False)

            if shingle_size is not None:
                trees = [rrcf.RCTree(points[ix].to_numpy(), index_labels=ix) for ix in indices]
            else:
                # [print(df_p.iloc[ix].to_numpy()) for ix in indices]
                trees = [rrcf.RCTree(df_p.iloc[ix].to_numpy(), index_labels=ix) for ix in indices]
            forest.extend(trees)

        # Calculate Codisp
        avg_codisp = pd.Series(0.0, index=np.arange(num_points))
        n_owning_trees = np.zeros(num_points)
        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(n_owning_trees, codisp.index.values, 1)
        avg_codisp /= n_owning_trees
    else:
        avg_codisp = {}
        streaming_points = df_p.to_numpy()
        for index, point in enumerate(streaming_points):
            for tree in forest:
                # If tree is above permitted size...
                if len(tree.leaves) > tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(index - tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute codisp on the new point...
                new_codisp = tree.codisp(index)
                # And take the average over all trees
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += new_codisp / num_trees

    # Save Codisp
    avg_codisp.to_json(sdp_directory / "codisp")


def main():
    test_pickle = 'NW000000000000000000000NBSNST0888'
    plot_dir = test_pickle + "phase_dif"
    # file_names = get_file_names(pickle_directory)
    # file_names = list(filter(lambda p: p in test_pickle, file_names))
    f_path = pickle_directory / test_pickle
    df_phases = list(
        map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)).loc[:, ['Value', 'phase_dif']], ['1', '2', '3']))
    df_p = df_phases[0].rename(columns={"Value":"p1"})
    df_p['p2'] = df_phases[1].Value
    df_p['p3'] = df_phases[2].Value
    # analyse_station_phase(df_p, plot_dir)
    s_codisp = pd.read_json(Path(f"../../rrcf/{plot_dir}/{codisp_filename}/codisp"), typ='series')
    old_string =f"Shingle{shingle_size}_TreeN{num_trees}_TreeS{tree_size}"
    ns_codisp = pd.read_json(Path(f"../../rrcf/{plot_dir}/{old_string}/codisp"), typ='series')
    plot_phase_dif(df_p, ns_codisp, s_codisp, plot_dir)


if __name__ == '__main__':
    main()
