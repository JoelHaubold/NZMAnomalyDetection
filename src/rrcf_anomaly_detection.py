import pandas as pd
from pandas import DataFrame
from src.anomaly_thresholds import AThreshold
import os
from pathlib import Path
import numpy as np
import rrcf
from multiprocessing import Process
import threading
from scipy import optimize


number_proc = 8
number_reps = 3

rrcf_cod_directory = Path("../../rrcf_cod")
rrcf_result_file = Path("./../results/rrcf_results")
pickle_directory = Path("../../pickles")
# num_trees = 200
# tree_size = 10000
# shingle_size = 0
# codisp_filename = f"Shingle{shingle_size}_TreeN{num_trees}_TreeS{tree_size}"

num_trees_opt = [100]  #[100, 150, 200]
tree_size_opt = [10000]#[5000, 10000, 15000, 20000]
shingle_size_opt = [0] #[0, 3, 5]

test_sample_file = Path("../test_sections.csv")


def get_file_names(file_directory):
    file_names = os.listdir(file_directory)
    file_names = list(filter(lambda f_path: os.path.isdir(file_directory / f_path), file_names))
    return file_names


def get_f_value(df_c: DataFrame, rel_cod_threshold, a_threshold: AThreshold):
    abs_codisp = df_c['codisp'].copy()
    nmbr_true_pos = df_c[(abs_codisp > rel_cod_threshold) & (df_c[a_threshold.name] > a_threshold.value)].shape[0]
    recall = nmbr_true_pos / df_c[df_c[a_threshold.name] > a_threshold.value].shape[0]
    precision = nmbr_true_pos / df_c[abs_codisp > rel_cod_threshold].shape[0]
    f1 = 2 * recall*precision / (recall+precision)
    f0_5 = 1.25 * recall*precision / (recall+(0.25*precision))
    f2 = 5 * recall*precision / (recall+(4*precision))
    return f1, f0_5, f2, precision, recall


def analyse_station_phase(df_p, df_constr, tree_size, nmbr_trees, shingle_size):
    # sdp_directory = rrcf_directory / station_name / codisp_filename
    # if not os.path.exists(sdp_directory):
    #     os.makedirs(sdp_directory)

    # Prepare points
    num_points = df_p.shape[0]
    print(num_points)
    if shingle_size != 0:
        points = rrcf.shingle(df_p, size=shingle_size)
        points = np.vstack([point for point in points])
        num_points = points.shape[0]
    sample_size_range = (num_points // tree_size, tree_size)
    forest = []

    # Construct forest
    avg_codisp = {}
    print(df_p)
    print(df_constr)
    streaming_points = df_p.to_numpy()
    constr_points = df_constr.to_numpy()
    print(streaming_points)
    print(constr_points)
    for _ in range(nmbr_trees):
        tree = rrcf.RCTree()
        forest.append(tree)

    for index, point in enumerate(constr_points):
        if index % 1000 == 0:
            print('c'+str(index))
        for tree in forest:
            tree.insert_point(point, index=index)

    for index, point in enumerate(streaming_points):
        if index % 1000 == 0:
            print(index)
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
    return avg_codisp


def calc_mcc(df_p, avg_codisp, cod_threshold, a_threshold):
    tp = df_p[(avg_codisp >= cod_threshold) & (df_p[a_threshold.name] >= a_threshold.value)].shape[0]
    tn = df_p[(avg_codisp < cod_threshold) & (df_p[a_threshold.name] < a_threshold.value)].shape[0]
    fp = df_p.shape[0] - tp
    fn = df_p.shape[0] - tn

    mcc = (tp*tn - fp*fn)/(((tp+fp)(tp+fn)(tn+fp)(tn+fn))**0.5)

    return mcc


def calc_opt_cod_threshold(df_p, avg_codisp, a_threshold):

    scipy_result = optimize.fmin(lambda x: -calc_mcc(df_p, avg_codisp, x, a_threshold), 100)
    opt_mcc = scipy_result.fopt
    cod_thresh = scipy_result.xopt
    return cod_thresh, opt_mcc


lock = threading.Lock()


def save_result(station, start_date, end_date, tree_size, nmbr_trees, shingle_size, cod_threshold, mcc, calc_nmbr):
    with lock:
        df = pd.read_pickle(rrcf_result_file)
        df = df.append({'station': station,'start_date': start_date,'end_date': end_date,'tree_size': tree_size,
                        'nmbr_trees': nmbr_trees, 'shingle_size': shingle_size, 'cod_threshold': cod_threshold,
                        'mcc': mcc, 'calc_nmbr': calc_nmbr}, ignore_index=True)
        df.to_pickle(rrcf_result_file)


def run_hp_test(df_constr, df_p, station, start_date, end_date, tree_size, nmbr_trees, shingle_size, a_type, calc_nmbr):
    avg_codisp = analyse_station_phase(df_p, df_constr, tree_size, nmbr_trees, shingle_size)
    np.save(arr=avg_codisp, file=rrcf_cod_directory / f"{station}_{start_date}_{end_date}_{tree_size}_{nmbr_trees}_{shingle_size}_{a_type.name}_{calc_nmbr}")

    cod_threshold, mcc = calc_opt_cod_threshold(df_p, avg_codisp, a_type)

    save_result(station, start_date, end_date, tree_size, nmbr_trees, shingle_size, cod_threshold, mcc, calc_nmbr)


def get_test_sample(station, start, end, a_type, tree_size):
    f_path = pickle_directory / station
    df_phases = list(
        map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)).loc[:, ['Value', 'phase_dif']], ['1', '2', '3']))
    df_p = df_phases[0].rename(columns={"Value": "p1"})
    df_p['p2'] = df_phases[1].Value
    df_p['p3'] = df_phases[2].Value

    df_p = df_p[[a_type.name]]

    start_row = df_p.index.get_loc(start, method='backfill').start
    df_const = df_p.iloc[(start_row - tree_size):start_row]

    df_p = df_p[start:end]

    return df_p, df_const


running_procs = []


def start_process(p):
    i = 0
    while len(running_procs) >= number_proc:
        if i <= 8:
            ex_code = running_procs[i].join(1)
            if ex_code is not None:
                del running_procs[i]
            else:
                i = i + 1
        else:
            i = 0

    running_procs.append(p)
    p.start()


def test_h_parameters():
    sample_df = pd.read_csv(test_sample_file)

    for index, row in sample_df.iterrows():
        for ts in tree_size_opt:
            for tn in num_trees_opt:
                for ss in shingle_size_opt:
                    station = row['station']
                    start = row['start']
                    end = row['end']
                    a_type = row['a_type']
                    a_type = AThreshold[a_type]
                    df_p, df_const = get_test_sample(station, start, end, a_type, ts)

                    for pn in range(number_reps):
                        calc_nmbr = np.random.randint(low=0, high=999999)
                        p = Process(target=run_hp_test, args=(df_const, df_p, station, start, end, ts, tn, ss, a_type, calc_nmbr))
                        start_process(p)


def main():
    test_h_parameters()


if __name__ == '__main__':
    main()