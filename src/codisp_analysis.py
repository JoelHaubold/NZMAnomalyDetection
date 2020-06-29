from pathlib import Path
import pandas as pd
import time
import numpy as np

from src.anomaly_thresholds import AThreshold
from multiprocessing import Process, Manager

test_sections = Path("./../test_sections.csv")
rrcf_result_file = Path("./../results")
rrcf_cod_directory = Path("../../rrcf_cod")
pickle_directory = Path("../../pickles")

number_proc = 16


def get_atype(row, df_dict):
    return df_dict.loc[df_dict['station'] == row['station']].iloc[0]['a_type']


def add_atype(df_extend):
    df_dict = pd.read_csv(test_sections)
    df_extend['a_type'] = df_extend.apply(lambda row: get_atype(row, df_dict), axis=1)
    return df_extend


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
    print(f"acc:{accuracy},thres:{cod_threshold}")
    return mcc, accuracy, tp, tn, fp, fn

def getAvg(prev_avg, x, n):
    return ((prev_avg * n + x) / (n + 1))

thres_opt = np.arange(10, 600 ,5)
accuracy_dict = dict.fromkeys(thres_opt, 0)
tp_dict = dict.fromkeys(thres_opt, 0)
tn_dict = dict.fromkeys(thres_opt, 0)
fp_dict = dict.fromkeys(thres_opt, 0)
fn_dict = dict.fromkeys(thres_opt, 0)


def calc_opt_cod_threshold(df_p, avg_codisp, a_threshold, shingle_size, nmbr_calc_accs):
    calculated_mcc = {}
    calculated_acc = {}
    max_mcc = -1
    max_acc = -1
    for to in thres_opt:
        mcco, acco, tp, tn, fp, fn = calc_mcc(df_p, avg_codisp, to, a_threshold, shingle_size)
        calculated_mcc[mcco] = to
        calculated_acc[to] = acco
        max_mcc = max(max_mcc, mcco)
        max_acc = max(max_acc, acco)
        accuracy_dict[to] = getAvg(accuracy_dict[to], acco, nmbr_calc_accs)
        tp_dict[to] = getAvg(tp_dict[to], tp, nmbr_calc_accs)
        tn_dict[to] = getAvg(tn_dict[to], tn, nmbr_calc_accs)
        fp_dict[to] = getAvg(fp_dict[to], fp, nmbr_calc_accs)
        fn_dict[to] = getAvg(fn_dict[to], fn, nmbr_calc_accs)
    return calculated_mcc[max_mcc], max_mcc, calculated_acc[calculated_mcc[max_mcc]]


def get_accuracy(index, row, t_s_max, nmbr_calc_accs):
    c_n = row['calc_nmbr']
    station = row['station']
    s_s = row['shingle_size']
    start_date = row['start_date']
    a_type = AThreshold[row['a_type']]
    phase = row['phase']

    f_path = pickle_directory / station
    df_p = pd.read_pickle(f_path / ("h_phase" + str(phase)))
    df_p = df_p[[a_type.name]]
    start_row = df_p.index.get_loc(start_date, method='backfill').start
    df_p = df_p.iloc[(start_row - s_s + 1):start_row + t_s_max + 1]
    avg_codisp = np.load(rrcf_cod_directory / (str(c_n) + ".npy"))

    x, y, acc = calc_opt_cod_threshold(df_p, avg_codisp, a_type, s_s, nmbr_calc_accs)

    return acc



def add_accuracy(df_extend):
    t_s_max = max(df_extend['tree_size'])
    # manager = Manager()
    # return_dict = manager.dict()
    acc_list = []
    nmbr_calc_accs = 0
    for index, row in df_extend.iterrows():
        acc = get_accuracy(index, row, t_s_max, nmbr_calc_accs)
        acc_list.append(acc)
        nmbr_calc_accs = nmbr_calc_accs + 1
        # p = Process(target=get_accuracy, args=(index, row, t_s_max, return_dict))
        # start_process(p)
    print("x")
    df_extend['accuracy']= acc_list
    print("x")


# dfptp = dfmean.agg(np.ptp)
# dfptp.groupby("nmbr_trees").mean()


def main():
    df_extend = pd.read_csv(rrcf_result_file / "rrcf_results_second_run.csv", index_col=0)
    df_extend = add_accuracy(df_extend)
    # df_extend.to_csv(rrcf_result_file / "rrcf_results_second_run2.csv")


if __name__ == '__main__':
    main()