import pandas as pd
from pandas import DataFrame
from src.anomaly_thresholds import AThreshold
import os
from pathlib import Path
import numpy as np
import rrcf
from multiprocessing import Process
import threading
import time
import sys
from pandas.plotting import table
import matplotlib.pyplot as plt
#  from scipy import optimize


number_proc = 16
number_reps = 1

rrcf_cod_directory = Path("../../rrcf_cod")
rrcf_result_file = Path("./../results/rrcf_results_phase_dif")
pickle_directory = Path("../../pickles")
# num_trees = 200
# tree_size = 10000
# shingle_size = 0
# codisp_filename = f"Shingle{shingle_size}_TreeN{num_trees}_TreeS{tree_size}"

num_trees_opt = [150]
tree_size_opt = [25000, 20000, 15000]
shingle_size_opt = [1]

test_sample_file = Path("../phasedif_test_sections.csv")


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


def analyse_station_phase(df_p, df_constr, tree_size, nmbr_trees, shingle_size, a_type):
    # sdp_directory = rrcf_directory / station_name / codisp_filename
    # if not os.path.exists(sdp_directory):
    #     os.makedirs(sdp_directory)

    streaming_points = df_p[a_type.name].values
    constr_points = df_constr[a_type.name].values
    # Prepare points
    if shingle_size != 1:
        points = rrcf.shingle(streaming_points, size=shingle_size)
        points = np.vstack([point for point in points])
        streaming_points = points
        points = rrcf.shingle(constr_points, size=shingle_size)
        points = np.vstack([point for point in points])
        constr_points = points
        # num_points = points.shape[0]

    # sample_size_range = (num_points // tree_size, tree_size)
    forest = []

    # Construct forest
    avg_codisp = {}

    for _ in range(nmbr_trees):
        cps = np.array(constr_points)
        if shingle_size==1:
            cps = np.vstack(cps)
        ixs = np.array(range(len(constr_points)))
        tree = rrcf.RCTree(cps, index_labels=ixs)
        forest.append(tree)

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


def calc_mcc(df_points, avg_codisp_a, cod_threshold, a_threshold, shingle_size):
    df_p = df_points.iloc[shingle_size-1:]
    avg_codisp = pd.Series(index=df_p.index, data=avg_codisp_a)
    tp = df_p[(avg_codisp >= cod_threshold) & (df_p[a_threshold.name] >= a_threshold.value)].shape[0]
    tn = df_p[(avg_codisp < cod_threshold) & (df_p[a_threshold.name] < a_threshold.value)].shape[0]
    fp = df_p[(avg_codisp >= cod_threshold) & (df_p[a_threshold.name] < a_threshold.value)].shape[0]
    fn = df_p[(avg_codisp < cod_threshold) & (df_p[a_threshold.name] >= a_threshold.value)].shape[0]

    accuracy = (tp+tn)/df_p.shape[0]

    if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0:
        mcc = (tp*tn - fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    else:
        mcc = tp*tn - fp*fn
    # print(f"mcc:{mcc},thres:{cod_threshold}")
    return mcc, accuracy, tp, tn, fp, fn


thres_opt = np.arange(10, 600 ,5)


def calc_opt_cod_threshold(df_p, avg_codisp, a_threshold, shingle_size):
    calculated_mcc = {}
    calculated_acc, calculated_tp, calculated_tn, calculated_fp, calculated_fn = {}, {}, {}, {}, {}
    max_mcc = -1
    # max_acc = -1
    acc120, tp120, tn120, fp120, fn120 = 0, 0, 0, 0, 0
    for to in thres_opt:
        mcco, acco, tp, tn, fp, fn = calc_mcc(df_p, avg_codisp, to, a_threshold, shingle_size)
        if to == 120:
            acc120 , tp120, tn120, fp120, fn120 = acco, tp, tn, fp, fn
        calculated_mcc[mcco] = to
        calculated_acc[to] = acco
        max_mcc = max(max_mcc, mcco)
        calculated_tp[to], calculated_tn[to], calculated_fp[to], calculated_fn[to] = tp, tn, fp, fn
        # max_acc = max(max_acc, acco)
    # cod_thresh, opt_mcc = optimize.fmin(func=lambda x: -calc_mcc(df_p, avg_codisp, int(x), a_threshold), x0=20)
    if max_mcc != 0:
        opt_to = calculated_mcc[max_mcc]
        acc_ret = calculated_acc[opt_to]
        tp_ret = calculated_tp[opt_to]
        tn_ret = calculated_tn[opt_to]
        fp_ret = calculated_fp[opt_to]
        fn_ret = calculated_fn[opt_to]
    else:
        acc_ret, tp_ret, tn_ret, fp_ret, fn_ret = acc120, tp120, tn120, fp120, fn120
    return calculated_mcc[max_mcc], max_mcc, acc_ret, tp_ret, tn_ret, fp_ret, fn_ret


lock = threading.Lock()


def save_result(station, start_date, end_date, tree_size, nmbr_trees, shingle_size, cod_threshold, mcc, acc, tp, tn, fp, fn, calc_nmbr, phase, a_type, t_constructed, t_needed):
    with lock:
        if rrcf_result_file.exists():
            df = pd.read_pickle(rrcf_result_file)
        else:
            df = pd.DataFrame(columns=['station', 'start_date', 'end_date', 'tree_size', 'nmbr_trees', 'shingle_size',
                                       'cod_threshold', 'mcc', 'acc', 'tp', 'tn', 'fp', 'fn', 'calc_nmbr', 'phase', 'a_name', 'c_time', 'r_time'])
        df = df.append({'station': station,'start_date': start_date, 'tree_size': tree_size,
                        'nmbr_trees': nmbr_trees, 'shingle_size': shingle_size, 'cod_threshold': cod_threshold,
                        'mcc': mcc, 'acc': acc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'calc_nmbr': calc_nmbr, 'phase': phase, 'a_name': a_type.name,
                        'c_time': t_constructed,'r_time': t_needed- t_constructed}, ignore_index=True)
        df.to_pickle(rrcf_result_file)


def run_hp_test(df_constr, df_p, station, start_date, end_date, tree_size, nmbr_trees, shingle_size, a_type, calc_nmbr, phase):
    # avg_codisp = np.load(file=rrcf_cod_directory / "512333.npy", allow_pickle=True)
    # print(avg_codisp)
    # cod_threshold, mcc = calc_opt_cod_threshold(df_p, avg_codisp, a_type)
    t_start = time.time()
    avg_codisp, t_constructed = analyse_station_phase(df_p, df_constr, tree_size, nmbr_trees, shingle_size, a_type)
    t_needed = (time.time() - t_start)/60
    t_constructed = (t_constructed - t_start)/60
    np.save(arr=avg_codisp, file=rrcf_cod_directory / str(calc_nmbr))
    print(f"CalcNmbr: {calc_nmbr} <-- {nmbr_trees}-{shingle_size}-{tree_size}-{a_type.name}")

    cod_threshold, mcc, acc, tp, tn, fp, fn = calc_opt_cod_threshold(df_p, avg_codisp, a_type, shingle_size)

    save_result(station, start_date, end_date, tree_size, nmbr_trees, shingle_size, cod_threshold, mcc, acc, tp, tn, fp, fn, calc_nmbr, phase, a_type, t_constructed, t_needed)


def get_test_sample(station, start, end, a_type, tree_size, shingle_size,phase):
    f_path = pickle_directory / station
    #df_phases = list(
    #    map(lambda p: pd.read_pickle(f_path / ("h_phase" + p)), ['1', '2', '3']))
    #df_p = df_phases[0].rename(columns={"Value": "p1"})
    #df_p['p2'] = df_phases[1].Value
    #df_p['p3'] = df_phases[2].Value
    df_p = pd.read_pickle(f_path / ("h_phase" + str(phase)))
    df_p = df_p[[a_type.name]]

    start_row = df_p.index.get_loc(start, method='backfill').start
    end_row = df_p.index.get_loc(end, method='pad').start
    df_const = df_p.iloc[(start_row - tree_size - shingle_size + 1):start_row]

    # df_p = df_p.iloc[(start_row - shingle_size + 1):end_row+1]
    tree_size_max = max(tree_size_opt)
    df_p = df_p.iloc[(start_row - shingle_size + 1):start_row + tree_size_max]

    return df_p, df_const


running_procs = []


def start_process(p):
    i = 0

    procs_to_use = number_proc
    try:
        nmbr_procs_file = open("./../nmbr_procs")
        nmbr_procs = int(nmbr_procs_file.read())
        print(f"Proceeding with: {nmbr_procs} procs")
        procs_to_use = number_proc

    except:
        print(f"reading number procs failed using: {number_proc}")



    while len(running_procs) >= procs_to_use:
        if i < procs_to_use:
            # ex_code = running_procs[i].join(1)
            if not running_procs[i].is_alive():
                del running_procs[i]
            else:
                i = i + 1
        else:
            i = 0
            time.sleep(6)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            # print(current_time)


    running_procs.append(p)
    p.start()


def test_h_parameters():
    sample_df = pd.read_csv(test_sample_file)

    for index, row in sample_df.iterrows():
        station = row['station']
        start = row['start']
        end = row['end']
        a_type = row['a_type']
        a_type = AThreshold[a_type]
        phase = row['phase']
        for tn in num_trees_opt:
            for ts in tree_size_opt:
                for ss in shingle_size_opt:
                    df_p, df_const = get_test_sample(station, start, end, a_type, ts, ss, phase)
                    for pn in range(number_reps):
                        calc_nmbr = np.random.randint(low=0, high=99999999)
                        p = Process(target=run_hp_test, args=(df_const, df_p, station, start, end, ts, tn, ss, a_type, calc_nmbr, phase))
                        start_process(p)
                        # time.sleep(1000000000)
    for rp in running_procs:
        c = 0
        while rp.is_alive():
            time.sleep(6)
            t = time.localtime()
            if c == 100:
                current_time = time.strftime("%H:%M:%S", t)
                print(current_time)
                c = 0
            c = c + 1

#  with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
def main():
    print(10**6)
    sys.setrecursionlimit(10**6)
#    while True:
#        print(time.time())
#        time.sleep(10)
    test_h_parameters()


if __name__ == '__main__':
    main()