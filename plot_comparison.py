#! /usr/bin/python3

# This script plots the comparison graph

import matplotlib.pyplot as plt

import sys
import pandas as pd


def collect_data():
    dsemble_best_file = "./output/gtsrb_label_err_dsemble_best"
    random_select_file = "./output/gtsrb_label_err_random_select"

    selected_fields = [1, 2, 4]

    dsemble_best = pd.read_csv(dsemble_best_file, header=None).tail(3).iloc[:,selected_fields]
    dsemble_best["Algorithm"] = "dsemble"
    random_select = pd.read_csv(random_select_file, header=None).iloc[:,selected_fields]
    random_select["Algorithm"] = "random"

    combined = pd.concat([dsemble_best, random_select], axis=0, ignore_index=True)
    combined.columns = ["Fault_Type", "Fault_Amt", "AD", "Algorithm"]
    combined["Fault_Type"] = combined["Fault_Type"].str.strip()
    combined = combined[["Fault_Type", "Fault_Amt", "Algorithm", "AD" ]]
    return combined


def main():
    benchmark = "gtsrb"
    fault_type = "label_err"

    fontsize = 16

    odf = collect_data()

    print(odf)

    odf = odf[odf["Fault_Type"] == fault_type]

    df = odf.pivot("Fault_Amt", "Algorithm", "AD")
    ax = df.plot(kind='bar', cmap="Paired")

    plt.xlabel("Fault Amount (%)", fontsize=fontsize)
    plt.ylabel("AD", fontsize=fontsize)
    
    plt.xticks(rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.yticks([0,0.1,0.2,0.3,0.4])

    bars = ax.patches
    df_width = len(df.columns.values)
    patterns = ['', 'O']

    hatches = []
    for h in range(df_width):  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / df_width)):
            hatches.append(patterns[h])

    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
        bar.set_alpha(0.99)

    ax.legend(["D-semble", "Random"], fontsize=fontsize, loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()

