import pandas as pd
import numpy as np
import parser
import loader
import sys
import time
from collections import defaultdict


def selinger_join_order(feature_dict, join_list, conditions_lst):
    join_order_lst = []

    return join_list



def left_deep_join(join_order_lst, root_path, conditions_lst):
    join_result = pd.DataFrame()
    joined_table = set()
    initialize = True

    for join_pair in join_order_lst:
        left_join_table = join_pair[0][0]
        left_join_col = join_pair[0][1]
        right_join_table = join_pair[1][0]
        right_join_col = join_pair[1][1]

        if initialize:
            join_result = initial_join_table( left_join_table, right_join_table,left_join_col, right_join_col,root_path, conditions_lst)



            if join_result.shape[0]==0:
                return join_result

            joined_table.add(left_join_table)
            joined_table.add(right_join_table)
            initialize = False

        else:

            if left_join_table in joined_table and right_join_table in joined_table:

                join_result = self_join(left_join_table, right_join_table, left_join_col, right_join_col, root_path, join_result)

                if join_result.shape[0]==0:
                    return join_result

            elif left_join_table in joined_table:

                join_result = join_table(left_join_table,right_join_table,  left_join_col, right_join_col,root_path, join_result,conditions_lst)

                if join_result.shape[0]==0:
                    return join_result

                joined_table.add(right_join_table)


            elif right_join_table in joined_table:
                join_result = join_table(right_join_table,left_join_table, right_join_col, left_join_col,root_path,join_result,conditions_lst)

                if join_result.shape[0]==0:
                    return join_result

                joined_table.add(left_join_table)


    return join_result


def self_join(left_join_table, right_join_table, left_join_col, right_join_col, root_path, join_result):
    temp_join_result = pd.DataFrame(columns=list(join_result.columns))


    with open(root_path + left_join_col + ".dat")  as left_file:
        left = pd.DataFrame(np.fromfile(left_file, dtype="int"))
    left_file.close()
    left.rename(columns={0: left_join_col}, inplace=True)

    left= left[left.index.isin(join_result[left_join_table])]


    with open(root_path + right_join_col + ".dat")  as right_file:
        right = pd.DataFrame(np.fromfile(right_file, dtype="int"))
    right_file.close()
    right.rename(columns={0: right_join_col}, inplace=True)

    right= right[right.index.isin(join_result[right_join_table])]


    for index, row in join_result.iterrows():
        left_index = row[left_join_table]
        right_index = row[right_join_table]

        if left.loc[left_index][left_join_col] == right.loc[right_index][right_join_col]:

            temp_join_result= temp_join_result.append(row)

    join_result = temp_join_result

    return join_result


def join_table(joined_table, waiting_join_table, joined_table_col, waiting_join_col, root_path, join_result, conditions_lst ):

    join_list = list(join_result.columns)
    join_list.append(waiting_join_table)

    temp_join_result = pd.DataFrame(columns=join_list)


    #  check title problem

    with open(root_path + joined_table_col + ".dat")  as joined_file:
        joined_df = pd.DataFrame(np.fromfile(joined_file, dtype="int"))
    joined_file.close()
    joined_df.rename(columns={0: joined_table_col}, inplace=True)

    joined_df = joined_df[joined_df.index.isin(join_result[joined_table])]


    with open(root_path + waiting_join_col + ".dat")  as waiting_join_file:
        waiting_join_df = pd.DataFrame(np.fromfile(waiting_join_file, dtype="int"))

    waiting_join_file.close()

    # waiting_join_df["idx"] = waiting_join_df.index
    waiting_join_df.rename(columns={0: waiting_join_col}, inplace=True)


    for condition in conditions_lst:
        if waiting_join_table == condition[0]:

            waiting_join_df = filter_condition(waiting_join_df, conditions_lst,root_path)

            break


    # join part

    if joined_df.shape[0] < waiting_join_df.shape[0]:

        joined_df_dict = dict()

        for joined_index, joined_row in joined_df.iterrows():
            row_value = joined_row[0]

            if row_value in joined_df_dict:
                joined_df_dict[row_value].append(joined_index)
            else:
                joined_df_dict[row_value]=[joined_index]


        for waiting_join_idx, waiting_join_row in waiting_join_df.iterrows():
            row_value = waiting_join_row[0]


            if row_value in joined_df_dict:

                for joined_index in joined_df_dict[row_value]:

                    temp_df = join_result[join_result[joined_table]==joined_index]

                    for idx, row in temp_df.iterrows():
                        row[waiting_join_table]= waiting_join_idx
                        temp_join_result = temp_join_result.append(row)


    else:
        waiting_join_df_dict = dict()

        for waiting_join_index, waiting_join_row in waiting_join_df.iterrows():
            row_value = waiting_join_row[0]

            if row_value in waiting_join_df_dict:
                waiting_join_df_dict[row_value].append(waiting_join_index)
            else:
                waiting_join_df_dict[row_value] = [waiting_join_index]


        for joined_idx, joined_row in joined_df.iterrows():
            row_value = joined_row[0]

            if row_value in waiting_join_df_dict:

                for waiting_join_index in waiting_join_df_dict[row_value]:

                    temp_df = join_result[join_result[joined_table] == joined_idx]

                    for idx, row in temp_df.iterrows():
                        row[waiting_join_table] = waiting_join_index
                        temp_join_result = temp_join_result.append(row)


    join_result = temp_join_result





    return join_result



def initial_join_table(left_join_table, right_join_table, left_join_col, right_join_col,root_path, conditions_lst):

    join_result = pd.DataFrame(columns=[left_join_table, right_join_table])


    with open(root_path + left_join_col + ".dat")  as left_file:
        left=pd.DataFrame(np.fromfile(left_file, dtype="int"))
    left_file.close()

    left.rename(columns={0: left_join_col}, inplace=True)


    for condition in conditions_lst:
        if left_join_table == condition[0]:
            left = filter_condition(left, conditions_lst,root_path)
            break



    with open(root_path + right_join_col + ".dat")  as right_file:
        right=pd.DataFrame(np.fromfile(right_file, dtype="int"))
    right_file.close()


    right.rename(columns={0: right_join_col}, inplace=True)



    for condition in conditions_lst:
        if right_join_table == condition[0]:
            right = filter_condition(right, conditions_lst,root_path)
            break


    if left.shape[0]< right.shape[0]:
        left_dict = dict()

        for left_index, left_row in left.iterrows():
            row_value = left_row[0]
            if row_value in left_dict:
                left_dict[row_value].append(left_index)
            else:
                left_dict[row_value]=[left_index]

        for right_idx, right_row in right.iterrows():
            right_value = right_row[0]

            if right_value in left_dict:
                for left_index in left_dict[right_value]:
                    join_result = join_result.append({left_join_table: left_index, right_join_table: right_idx},ignore_index=True)

    else:

        right_dict = dict()

        for right_index, right_row in right.iterrows():
            row_value = right_row[0]
            if row_value in right_dict:
                right_dict[row_value].append(right_index)
            else:
                right_dict[row_value] = [right_index]

        for left_idx, left_row in left.iterrows():
            left_value = left_row[0]

            if left_value in right_dict:
                for right_index in right_dict[left_value]:
                    join_result = join_result.append({left_join_table: left_idx, right_join_table: right_index},
                                                     ignore_index=True)





    return join_result



def filter_condition(df, conditions_list, root_path):


    for condition in conditions_list:
        condition_table = condition[0]
        condition_column = condition[1]
        condition_relation = condition[2]
        comparison_num = condition[3]

        if condition_table == df.columns[0][0] and df.columns[0] == condition_column:


            if condition_relation == "=":
                df = df[df[condition_column] == comparison_num]
            elif condition_relation == "<":
                df = df[df[condition_column] < comparison_num]

            elif condition_relation == ">":
                df = df[df[condition_column] > comparison_num]


        elif condition_table == df.columns[0][0]:

            with open(root_path + condition_column + ".dat")  as filter_file:
                filter_df = pd.DataFrame(np.fromfile(filter_file, dtype="int"))
            filter_file.close()

            # filter_df["idx"] = filter_df.index
            filter_df.rename(columns={0: condition_column}, inplace=True)

            filter_df = filter_df[filter_df.index.isin(df.index)]

            if condition_relation == "=":
                filter_df = filter_df[filter_df[condition_column] == comparison_num]
            elif condition_relation == "<":
                filter_df = filter_df[filter_df[condition_column] < comparison_num]

            elif condition_relation == ">":
                filter_df = filter_df[filter_df[condition_column] > comparison_num]

            df = df[df.index.isin(filter_df.index)]


    return df


# def filter_condition(df, conditions_list, root_path):
#
#
#     for condition in conditions_list:
#         condition_table = condition[0]
#         condition_column = condition[1]
#         condition_relation = condition[2]
#         comparison_num = condition[3]
#
#         temp_df = pd.DataFrame()
#
#         if condition_table == df.columns[0][0] and df.columns[0] == condition_column:
#
#
#             if condition_relation == "=":
#                 for index, row in df.iterrows():
#                     if row[0]== comparison_num:
#                         temp_df = temp_df.append(row)
#
#
#             elif condition_relation == "<":
#                 for index, row in df.iterrows():
#                     if row[0]< comparison_num:
#                         temp_df = temp_df.append(row)
#
#             elif condition_relation == ">":
#                 for index, row in df.iterrows():
#                     if row[0]> comparison_num:
#                         temp_df = temp_df.append(row)
#
#             df = temp_df
#
#
#         elif condition_table == df.columns[0][0]:
#
#             with open(root_path + condition_column + ".dat")  as filter_file:
#                 filter_df = pd.DataFrame(np.fromfile(filter_file, dtype="int"))
#             filter_file.close()
#
#             # filter_df["idx"] = filter_df.index
#             filter_df.rename(columns={0: condition_column}, inplace=True)
#
#             filter_df["idx"] = filter_df.index
#             df["idx"] = df.index
#
#             df_cardinality = df.shape[0]
#             filter_df_cardinality = filter_df.shape[0]
#             left_pointer =0
#             right_pointer = 0
#
#
#
#             if condition_relation == "=":
#
#
#                 while left_pointer < df_cardinality and right_pointer < filter_df_cardinality:
#                     if df.iloc[left_pointer]["idx"] == filter_df.iloc[right_pointer]["idx"]:
#                         if filter_df.iloc[right_pointer][condition_column] == comparison_num:
#                             temp_df = temp_df.append(df.iloc[left_pointer])
#
#                         left_pointer+=1
#                         right_pointer+=1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] > filter_df.iloc[right_pointer]["idx"]:
#                         right_pointer+=1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] < filter_df.iloc[right_pointer]["idx"]:
#                         left_pointer+=1
#                         continue
#
#             elif condition_relation == "<":
#
#                 while left_pointer < df_cardinality and right_pointer < filter_df_cardinality:
#                     if df.iloc[left_pointer]["idx"] == filter_df.iloc[right_pointer]["idx"]:
#                         if filter_df.iloc[right_pointer][condition_column] < comparison_num:
#                             temp_df = temp_df.append(df.iloc[left_pointer])
#
#                         left_pointer += 1
#                         right_pointer += 1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] > filter_df.iloc[right_pointer]["idx"]:
#                         right_pointer += 1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] < filter_df.iloc[right_pointer]["idx"]:
#                         left_pointer += 1
#                         continue
#
#             elif condition_relation == ">":
#                 while left_pointer < df_cardinality and right_pointer < filter_df_cardinality:
#                     if df.iloc[left_pointer]["idx"] == filter_df.iloc[right_pointer]["idx"]:
#                         if filter_df.iloc[right_pointer][condition_column] > comparison_num:
#                             temp_df = temp_df.append(df.iloc[left_pointer])
#
#                         left_pointer += 1
#                         right_pointer += 1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] > filter_df.iloc[right_pointer]["idx"]:
#                         right_pointer += 1
#                         continue
#
#                     if df.iloc[left_pointer]["idx"] < filter_df.iloc[right_pointer]["idx"]:
#                         left_pointer += 1
#                         continue
#             df = temp_df
#
#     return df



def get_sum_result_lst(join_result, select_list, root_path):
    result_list = []



    if join_result.shape[0]==0:
        return result_list

    for select in select_list:
        temp_sum=0

        with open(root_path + select + ".dat")  as sum_file:
            sum_df = pd.DataFrame(np.fromfile(sum_file, dtype="int"))
        sum_file.close()
        # sum_df["idx"] = sum_df.index
        sum_df.rename(columns={0: select}, inplace=True)


        sum_df = sum_df[sum_df.index.isin(join_result[select[0]])]




        for index , row in join_result.iterrows():
            temp_sum += sum_df.loc[row[select[0]]]


        result_list.append(int(temp_sum))


    return result_list



def print_string(join_result, result_list, select_list):

    if join_result.shape[0]==0:
        for i in range(len(select_list)-1):
            print(',', end='')
    else:

        for i in range(len(result_list)):
            print(result_list[i], end='')

            if i != len(result_list) - 1:
                print(',', end='')
    print()



def excutor():

    f = sys.stdin
    file_list_string = f.readline().rstrip('\n')

    num_of_sql = int(f.readline().rstrip('\n'))

    feature_dict=loader.execute_loader(file_list_string)

    root_path=loader.get_root_path(file_list_string)



    for i in range(num_of_sql):
        select_list = parser.parse_select(f.readline())
        f.readline()
        join_list = parser.parse_join(f.readline())
        conditions_list = parser.parse_condition(f.readline())
        f.readline()

        join_order_lst = selinger_join_order(feature_dict, join_list, conditions_list)

        join_result=left_deep_join(join_order_lst, root_path, conditions_list)

        result_list = get_sum_result_lst(join_result, select_list, root_path)

        print_string(join_result, result_list, select_list)



if __name__ == "__main__":

    # start_time = time.time()
    #
    # feature_dict = loader.execute_loader(
    #     "data/xxxs/A.csv,data/xxxs/B.csv,data/xxxs/C.csv,data/xxxs/D.csv,data/xxxs/E.csv")
    #
    # root_path = loader.get_root_path("data/xxxs/A.csv,data/xxxs/B.csv,data/xxxs/C.csv,data/xxxs/D.csv,data/xxxs/E.csv")
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # middle_time = time.time()
    #
    # with open("data/xxxs/queries.sql") as sql:
    #     sql_list = sql.readlines()
    #
    # for i in range(1, len(sql_list)):
    #
    #     if i % 5 == 1:
    #         select_list = parser.parse_select(sql_list[i - 1].rstrip('\n'))
    #         print(select_list)
    #     if i % 5 == 3:
    #         join_list = parser.parse_join(sql_list[i - 1].rstrip('\n'))
    #         print(join_list)
    #     if i % 5 == 4:
    #         conditions_list = parser.parse_condition(sql_list[i - 1].rstrip('\n'))
    #         print(conditions_list)
    #
    #     if i > 1 and i % 5 == 0:
    #         join_order_lst = selinger_join_order(feature_dict, join_list, conditions_list)
    #
    #         join_result = left_deep_join(join_order_lst, root_path, conditions_list)
    #
    #         result_list = get_sum_result_lst(join_result, select_list, root_path)
    #
    #         print_string(join_result, result_list, select_list)
    #
    #
    #
    #
    # print("--- %s seconds ---" % (time.time() - middle_time))

    excutor







