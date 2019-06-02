import pandas as pd
import numpy as np
import csv
from os import listdir
import glob
import struct
import os
from pathlib import Path
import time
from collections import defaultdict


def get_all_csv_files(file_list_string):

    return file_list_string.split(",")

def get_root_path(file_list_string):
    # split_lst=file_list_string.split("/")
    # root_path = split_lst[0]+"/"+split_lst[1]+"/"
    #
    # return root_path
    return "./"

def build_dat_file(csv_file, root_path):
    with open(csv_file) as f:
        reader = csv.reader(f)
        row1 = next(reader)

    lst = [None] * len(row1)

    for i in range(len(row1)):
        lst[i] = open(root_path + csv_file[-5:-3] + "c" + str(i) + ".dat", "wb")

    return lst



def load_data_from_csv(csv_file, lst, feature_dict):

    file_name = csv_file[-5]


    for chunk in pd.read_csv(csv_file, chunksize=500000000000,header=None):

        for i in range(len(lst)):

            col_name = file_name+".c" + str(i)

            chunk[i].values.tofile(lst[i])


            max = chunk[i].max()
            min = chunk[i].min()

            if max > feature_dict[col_name][0]:
                feature_dict[col_name][0] = max

            if min < feature_dict[col_name][1]:
                feature_dict[col_name][1] = min

            feature_dict[col_name][2]+= chunk[i].shape[0]



def execute_loader(file_list_string):
    file_list = get_all_csv_files(file_list_string)
    root_path = get_root_path(file_list_string)

    feature_dict  = defaultdict(lambda : [float('-inf'), float('inf'), 0])



    for csv_file in file_list:

        lst=build_dat_file(csv_file, root_path)


        load_data_from_csv(csv_file,lst, feature_dict)



    return feature_dict




if __name__ == "__main__":
    pass







