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
    labels = IsolationForest(contamination=0.0016).fit_predict(X)
    print(labels)
    mcc, accuracy, tp, tn, fp, fn = calc_mcc(df_p, labels, a_type)
    print(mcc, accuracy, tp, tn, fp, fn)
    return mcc


def get_test_sample(station, start, a_type, phase, run_window):
    f_path = pickle_directory / station
    df_p = pd.read_pickle(f_path / ("h_phase" + str(phase)))
    df_p = df_p[[a_type.name]]

    start_row = df_p.index.get_loc(start, method='backfill').start


    # df_p = df_p.iloc[(start_row - shingle_size + 1):end_row+1]
    tree_size_max = run_window  # max(tree_size_opt)
    df_p = df_p.iloc[start_row:start_row + tree_size_max]

    return df_p
mcc_list = []


def main():
    test_sample_file = Path("../trafodif_test_sections.csv")
    sample_df = pd.read_csv(test_sample_file)
    for index, row in sample_df.iterrows():
        a_type = AThreshold["trafo"]
        df_p = get_test_sample(row["station"], row['start'], a_type, 1, 25000)
        mcc = run_test(df_p, a_type)
        mcc_list.append(mcc)
    s = pd.Series(data=mcc_list)
    mcc_average = s.mean()
    print(mcc_average)
    print("x")


if __name__ == '__main__':
    main()

# phase = 0.14750792503444723 0.3338373086485978
# trafo = 0.08833066458127181 0.718359376599812
# time =  0.1806321374521998 0.342480058655457
# station = 0.20415526519894023 0.20308829189867555
# seas = 0.1066211871334278 0.6692231027794284