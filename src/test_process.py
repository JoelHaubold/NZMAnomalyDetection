import time
from pathlib import Path

import rrcf
import numpy as np
from src.anomaly_thresholds import AThreshold
import pandas as pd


pickle_directory = Path("../../pickles")


class TestParam:
    def __init__(self, tree_size_opt, run_window, num_trees_opt, shingle_size_opt, test_sample_file, result_file, reps):
        self.tree_size_opt = tree_size_opt
        self.run_window = run_window
        self.num_trees_opt = num_trees_opt
        self.shingle_size_opt = shingle_size_opt
        self.test_sample_file = test_sample_file
        self.result_file = result_file
        self.reps = reps


def get_test_sample(station, start, end, a_type, tree_size, shingle_size, phase, run_window):
    f_path = pickle_directory / station
    # df_phases = list(
    #    map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)), ['1', '2', '3']))
    # df_p = df_phases[0].rename(columns={"Value": "p1"})
    # df_p['p2'] = df_phases[1].Value
    # df_p['p3'] = df_phases[2].Value
    df_p = pd.read_pickle(f_path / ("h_phase" + str(phase)))
    # print(df_p.dtypes)

    # df_a = df_p[[AThreshold["phase_dif"].name]]
    # print(df_a)
    # print(df_a.dtypes)

    df_p = df_p[[a_type.name]]

    # print(df_p)
    # print(df_p.dtypes)
    start_row = df_p.index.get_loc(start, method='backfill').start
    # end_row = df_p.index.get_loc(end, method='pad').start
    df_const = df_p.iloc[(start_row - tree_size - shingle_size + 1):start_row]

    # df_p = df_p.iloc[(start_row - shingle_size + 1):end_row+1]
    tree_size_max = run_window  # max(tree_size_opt)
    df_p = df_p.iloc[(start_row - shingle_size + 1):start_row + tree_size_max]

    return df_p.copy(), df_const.copy()


def analyse_station_phase(df_p, df_constr, tree_size, nmbr_trees, shingle_size, a_type):
    # sdp_directory = rrcf_directory / station_name / codisp_filename
    # if not os.path.exists(sdp_directory):
    #     os.makedirs(sdp_directory)
    print(f"{nmbr_trees}-{shingle_size}-{tree_size}-{a_type.name}: c")
    streaming_points = df_p[a_type.name].values
    constr_points = df_constr[a_type.name].values
    streaming_points = abs(streaming_points)
    streaming_points = np.around(streaming_points, 3)
    constr_points = abs(constr_points)
    constr_points = np.around(constr_points, 3)
    # Prepare points
    # if shingle_size != 1:
    #     points = rrcf.shingle(streaming_points, size=shingle_size)
    #     points = np.vstack([point for point in points])
    #     streaming_points = points
    #     points = rrcf.shingle(constr_points, size=shingle_size)
    #     points = np.vstack([point for point in points])
    #     constr_points = points
    #     # num_points = points.shape[0]

    # sample_size_range = (num_points // tree_size, tree_size)
    forest = []

    # Construct forest
    avg_codisp = {}
    cps = np.array(constr_points)
    if shingle_size == 1:
        cps = np.vstack(cps)
    # print(cps.dtype)
    # print(len(cps))
    # cps = abs(cps)
    # print(cps)
    tim = time.time()
    for _ in range(nmbr_trees):
        ixs = np.array(range(len(constr_points)))
        print(time.time() - tim)
        tree = rrcf.RCTree(cps, index_labels=ixs)
        print(time.time() - tim)
        forest.append(tree)
        print(time.time() - tim)

    # for _ in range(nmbr_trees):
    #     tree = rrcf.RCTree()
    #     forest.append(tree)
    # for index, point in enumerate(constr_points):
    #     if index % 1000 == 0:
    #         print('c'+str(index))
    #     for tree in forest:
    #         tree.insert_point(point, index=index)
    t_constructed = time.time()
    for index, point in enumerate(streaming_points):
        if index % 5000 == 0:
            print(f"{nmbr_trees}-{shingle_size}-{tree_size}-{a_type.name}: {index}")
        index = index + tree_size
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
            if index not in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += new_codisp / nmbr_trees

    # Save Codisp
    # avg_codisp.to_json(sdp_directory / "codisp")
    # print(f"length codisp: {len(list(avg_codisp.values()))}")
    return list(avg_codisp.values()), t_constructed


def test_process():
    station = "NW000000000000000000000NBSNST0888"
    start = "2017-05-01 00:00:00"
    end = ""
    a_type = AThreshold["StationDif"]
    a_type2 = AThreshold["phase_dif"]
    ts = 25000
    tn = 150
    run_window = 25000
    ss = 1
    phase = 1
    df_p, df_const = get_test_sample(station, start, end, a_type, ts, ss, phase, run_window)
    # df_p2, df_const2 = get_test_sample(station, start, end, a_type2, ts, ss, phase, run_window)
    a, b = analyse_station_phase(df_p, df_const, ts, tn, ss, a_type)
    print("x")


def main():
    tp1 = TestParam([15000, 20000, 25000], 25000, [150], [1], Path("../stationdif_test_sections_err.csv"),
                    Path("./../results/rrcf_results_station_dif"), 1)
    test_process()


if __name__ == '__main__':
    main()
