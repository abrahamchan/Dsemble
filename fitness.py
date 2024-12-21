#! /usr/bin/python3

import itertools as IT

import pandas as pd
import numpy as np
import math
from itertools import combinations
import statistics

ens_size = 3

def get_eh(df):
    z = np.array(df, 'float')
    z = df / df.sum(axis=1).to_numpy()[:,np.newaxis]
    div = np.apply_along_axis(shannon, 1, z)
    return div

def shannon(y):
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    n = len(y)
    t = t[t!=0]
    H = -np.sum( t*np.log(t) )
    return H/np.log(ens_size)


def get_avg_eh(df, ensemble_size):
    temp_df = df.iloc[:,:ensemble_size].apply(pd.Series.value_counts, axis=1)
    #print(df.iloc[:,:ensemble_size])
    temp_df = temp_df.fillna(0)
    eh_arr = get_eh(temp_df)
    return statistics.mean(eh_arr)


def twoset(iterable):
    """twoset([A,B,C]) --> (A,B) (A,C) (B,C)"""
    s = list(iterable)
    return IT.chain.from_iterable(
        IT.combinations(s, r) for r in range(2,3))


def dict_twoset_counts(df):
    """twoset([A,B,C]) --> (A,B) (A,C) (B,C)"""
    """twoset_counts = {"A":{"B": [A==0 & B==0, A==0 & B==1, A==1 & B==0, A==1 & B==1],
                             "C": [A==0 & C==0, A==0 & C==1, A==1 & C==0, A==1 & C==1]},
                        "B":{"C": [B==0 & C==0, B==0 & C==1, B==1 & C==0, B==1 & C==1]}}"""
    result = {}
    for cols in twoset(df.columns):
        if not cols: continue
        result.setdefault(cols[0],{})
        for vals in IT.product([0,1], repeat=len(cols)):
            mask = np.logical_and.reduce([df[c]==v for c, v in zip(cols, vals)])
            cond = ' & '.join(['{}={}'.format(c,v) for c, v in zip(cols,vals)])
            n = len(df[mask])
            result[cols[0]].setdefault(cols[1],[]).append(n)
    return result

def get_disagreement_measure(df):
    temp_df = df.iloc[:,:-1]

    res = dict_twoset_counts(temp_df)

    col_list = list(df)[:-1]
    two_pairs = list(IT.combinations(col_list, 2))

    L = len(two_pairs)
    dis = 0.

    for mpair in two_pairs:
        m1 = mpair[0]
        m2 = mpair[1]
        n00 = res[m1][m2][0]
        n01 = res[m1][m2][1]
        n10 = res[m1][m2][2]
        n11 = res[m1][m2][3]

        dis += (n01 + n10) / (n11 + n10 + n01 + n00)

    dis /= L
    return dis

