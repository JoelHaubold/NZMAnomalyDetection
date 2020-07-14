from sklearn.ensemble import IsolationForest
import pandas as pd
from pathlib import Path
from src.anomaly_thresholds import AThreshold
import os
import numpy as np


pickle_directory = Path("../../pickles")


def calc_mcc(df_points, avg_codisp_a, a_threshold):
    df_p = df_points.iloc[0:]
    avg_codisp = pd.Series(index=df_p.index, data=avg_codisp_a)
    tp = df_p[(avg_codisp == -1) & (abs(df_p[a_threshold.name]) >= a_threshold.value)].shape[0]
    tn = df_p[(avg_codisp != -1) & (abs(df_p[a_threshold.name]) < a_threshold.value)].shape[0]
    fp = df_p[(avg_codisp == -1) & (abs(df_p[a_threshold.name]) < a_threshold.value)].shape[0]
    fn = df_p[(avg_codisp != -1) & (abs(df_p[a_threshold.name]) >= a_threshold.value)].shape[0]

    accuracy = (tp+tn)/df_p.shape[0]

    if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0:
        mcc = (tp*tn - fp*fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
    else:
        mcc = tp*tn - fp*fn
    # print(f"mcc:{mcc},thres:{cod_threshold}")
    return mcc, accuracy, tp, tn, fp, fn


def run_test(df_p, a_type):
    X = df_p.values
    conta = df_p[(abs(df_p[a_type.name]) >= a_type.value)].shape[0] / df_p.shape[0]
    print("predicting")
    labels = IsolationForest().fit_predict(X)
    print(labels)
    mcc, accuracy, tp, tn, fp, fn = calc_mcc(df_p, labels, a_type)
    print(mcc, accuracy, tp, tn, fp, fn)
    print("x")


def get_test_sample(station, start, a_type, phase, run_window):
    f_path = pickle_directory / station
    df_p = pd.read_pickle(f_path / ("h_phase" + str(phase)))
    df_p = df_p[[a_type.name]]

    start_row = df_p.index.get_loc(start, method='backfill').start


    # df_p = df_p.iloc[(start_row - shingle_size + 1):end_row+1]
    tree_size_max = run_window  # max(tree_size_opt)
    df_p = df_p.iloc[start_row:start_row + tree_size_max]

    return df_p


def main():
    a_type = AThreshold["phase_dif"]
    df_p = get_test_sample("NW000000000000000000000NBSNST0888", "2017-05-01 00:00:00", a_type, 1, 25000)
    run_test(df_p, a_type)


if __name__ == '__main__':
    main()