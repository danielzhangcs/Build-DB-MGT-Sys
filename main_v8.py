import numpy as np
import parser
import loader
import sys
import time
from collections import defaultdict
import copy
import bisect
from operator import itemgetter


class KeyList(object):
    # bisect doesn't accept a key function, so I build the key into the sequence.
    def __init__(self, l, key):
        self.l = l
        self.key = key
    def __len__(self):
        return len(self.l)
    def __getitem__(self, index):
        return self.key(self.l[index])


# estimate the cardinality after apply all filter condition

def estimate_cardinality_after_filter(temp_feature_dict, conditions_lst):

    for condition in conditions_lst:
        condition_table = condition[0]
        condition_column = condition[1]
        condition_relation = condition[2]
        comparison_num = condition[3]

        original_cardinality = temp_feature_dict[condition_column][2]
        original_max = temp_feature_dict[condition_column][0]
        original_min = temp_feature_dict[condition_column][1]
        original_unique = original_max - original_min

        if condition_relation == "=":
            temp_feature_dict[condition_column] = [comparison_num, comparison_num,
                                                   original_cardinality / original_unique, 1 / original_unique]
        if condition_relation == ">":
            temp_feature_dict[condition_column] = [original_max, comparison_num,
                                                   original_cardinality * (original_max - comparison_num) / (
                                                               original_max - original_min),
                                                   (original_max - comparison_num) / (original_max - original_min)]
        if condition_relation == "<":
            temp_feature_dict[condition_column] = [comparison_num, original_min,
                                                   original_cardinality * (comparison_num - original_min) / (
                                                           original_max - original_min),
                                                   (comparison_num - original_min) / (original_max - original_min)]
    return temp_feature_dict


# initialize the hash tale which include the best 2 element order and join cost
def selinger_join_order(feature_dict, temp_feature_dict, join_list, rels, conditions_lst):
    best = dict()

    for i in range(len(rels) - 1):
        for j in range(i + 1, len(rels)):
            helper = True
            left = rels[i]
            right = rels[j]

            left_cardinality = 1
            right_cardinality = 1
            left_remaining_size = 1
            right_remaining_size = 1

            for condition in conditions_lst:
                if left == condition[0] and left_cardinality == 1:
                    left_cardinality = temp_feature_dict[condition[1]][2]
                    left_remaining_size = temp_feature_dict[condition[1]][3]

                elif left == condition[0]:
                    left_cardinality = temp_feature_dict[condition[1]][2] * left_cardinality / \
                                       feature_dict[condition[1]][2]
                    left_remaining_size = temp_feature_dict[condition[1]][3] * left_remaining_size

                if right == condition[0] and right_cardinality == 1:
                    right_cardinality = temp_feature_dict[condition[1]][2]
                    right_remaining_size = temp_feature_dict[condition[1]][3]

                elif right == condition[0]:
                    right_cardinality = temp_feature_dict[condition[1]][2] * right_cardinality / \
                                        feature_dict[condition[1]][2]
                    right_remaining_size = temp_feature_dict[condition[1]][3] * right_remaining_size

            for join_pair in join_list:

                join_table_lst = [x[0] for x in join_pair]
                left_join_col = join_pair[0][1]
                right_join_col = join_pair[1][1]

                if left in join_table_lst and right in join_table_lst:
                    helper = False

                    if left_cardinality == 1:
                        left_cardinality = feature_dict[left_join_col][2]
                    if right_cardinality == 1:
                        right_cardinality = feature_dict[right_join_col][2]

                    left_unique = (feature_dict[left_join_col][0] - feature_dict[left_join_col][
                        1] + 1) * left_remaining_size
                    right_unique = (feature_dict[right_join_col][0] - feature_dict[right_join_col][
                        1] + 1) * right_remaining_size

                    remaining_size_after_join = min(left_unique, right_unique) / max(left_unique, right_unique)

                    estimated_after_join_cardinality = left_cardinality * right_cardinality * min(left_unique,
                                                                                                  right_unique) * (
                                                                   1 / (left_unique * right_unique))

                    if left_unique < right_unique:
                        best[(left, right)] = [[left, right],
                                               [left_remaining_size, remaining_size_after_join * right_remaining_size],
                                               left_cardinality + right_cardinality, estimated_after_join_cardinality]

                    else:
                        best[(left, right)] = [[left, right],
                                               [remaining_size_after_join * left_remaining_size, right_remaining_size],
                                               left_cardinality + right_cardinality, estimated_after_join_cardinality]
            if helper:
                best[(left, right)] = [[left, right], [1, 1], float('inf'), float('inf')]

    return best



# the selinger DP algorithm which will outpu the best join order

def compute_best(rels, best, temp_feature_dict, join_list, feature_dict, conditions_lst):
    if tuple(sorted(rels)) in best:
        return best[tuple(sorted(rels))]

    curr = [rels, [1 for x in rels], float('inf'), float('inf')]

    for i in rels:
        internal_best_order = compute_best([x for x in rels if x != i], best, temp_feature_dict, join_list,
                                           feature_dict, conditions_lst)

        waiting_join_table = i
        valid_join_pair_lst = []

        if internal_best_order[2] == float('inf'):
            continue

        for joined_table in internal_best_order[0]:

            for join_pair in join_list:

                join_table_lst = [x[0] for x in join_pair]
                left_join_col = join_pair[0][1]
                right_join_col = join_pair[1][1]
                valid_col_pair = [left_join_col, right_join_col]

                if joined_table in join_table_lst and waiting_join_table in join_table_lst:
                    valid_join_pair_lst.append([[x for x in valid_col_pair if joined_table in x][0],
                                                [x for x in valid_col_pair if waiting_join_table in x][0]])

        if len(valid_join_pair_lst) == 0:
            continue
        else:

            temp_curr = []
            initialize = True

            for valid_join_pair in valid_join_pair_lst:

                joined_table = valid_join_pair[0][0]
                waiting_join_table = valid_join_pair[1][0]

                joined_col = valid_join_pair[0]

                waiting_join_col = valid_join_pair[1]

                joined_table_index = internal_best_order[0].index(joined_table)

                joined_table_original_unique = feature_dict[joined_col][0] - feature_dict[joined_col][1] + 1

                if initialize:

                    initialize = False

                    joined_table_remaining_size = internal_best_order[1][joined_table_index]

                    waiting_join_table_remaining_size = 1

                    waiting_join_cardinality = 1

                    for condition in conditions_lst:
                        if waiting_join_table == condition[0] and waiting_join_cardinality == 1:
                            waiting_join_table_remaining_size = waiting_join_table_remaining_size * \
                                                                temp_feature_dict[condition[1]][3]

                            waiting_join_cardinality = temp_feature_dict[condition[1]][2]

                        elif waiting_join_table == condition[0]:
                            waiting_join_table_remaining_size = waiting_join_table_remaining_size * \
                                                                temp_feature_dict[condition[1]][3]

                            waiting_join_cardinality = temp_feature_dict[condition[1]][2] * waiting_join_cardinality / \
                                                       feature_dict[condition[1]][2]

                    if waiting_join_cardinality == 1:
                        waiting_join_cardinality = feature_dict[waiting_join_col][2]

                    curr_joined_table_unique = joined_table_original_unique * joined_table_remaining_size

                    cost = internal_best_order[2] + len(valid_join_pair_lst) * internal_best_order[
                        3] + waiting_join_cardinality

                    waiting_join_table_unique = (feature_dict[waiting_join_col][0] - feature_dict[waiting_join_col][
                        1] + 1) * waiting_join_table_remaining_size

                    remaining_size_after_join = min(curr_joined_table_unique, waiting_join_table_unique) / max(
                        curr_joined_table_unique, waiting_join_table_unique)

                    estimated_cardinality = internal_best_order[3] * waiting_join_cardinality * min(
                        curr_joined_table_unique, waiting_join_table_unique) * (
                                                        1 / (curr_joined_table_unique * waiting_join_table_unique))

                    if curr_joined_table_unique < waiting_join_table_unique:

                        temp_curr = [internal_best_order[0] + [waiting_join_table], internal_best_order[1] + [
                            remaining_size_after_join * waiting_join_table_remaining_size], cost, estimated_cardinality]



                    else:

                        temp_curr = [internal_best_order[0] + [waiting_join_table],
                                     [x * remaining_size_after_join for x in internal_best_order[1]] + [
                                         waiting_join_table_remaining_size], cost, estimated_cardinality]



                else:

                    waiting_join_table_unique = (feature_dict[waiting_join_col][0] - feature_dict[waiting_join_col][
                        1] + 1) * temp_curr[1][-1]

                    joined_table_unique = joined_table_original_unique * temp_curr[1][joined_table_index]

                    remaining_size_after_join = min(joined_table_unique, waiting_join_table_unique) / max(
                        joined_table_unique, waiting_join_table_unique)

                    temp_curr[1] = [x * remaining_size_after_join for x in temp_curr[1]]

                    temp_curr[3] = temp_curr[3] * remaining_size_after_join

        if temp_curr[2] < curr[2]:
            curr = temp_curr

    best[tuple(sorted(rels))] = curr

    return curr


# using the selinger output to find the best join pair
def find_best_order_pair(join_order, join_list):
    ordered_join_lst = []

    i = 0
    while i < len(join_order):
        left_join_table_lst = []

        if i == 0:
            left_join_table = join_order[i]
            right_join_table = join_order[i + 1]

            for join_pair in join_list:
                join_table_lst = [x[0] for x in join_pair]

                if left_join_table in join_table_lst and right_join_table in join_table_lst:
                    correct_order_pair = tuple([[x for x in join_pair if x[0] == left_join_table][0],
                                                [x for x in join_pair if x[0] == right_join_table][0]])
                    ordered_join_lst.append(correct_order_pair)
            i += 2
        else:
            multiple_join_pair = []

            for index in range(i):
                left_join_table_lst.append(join_order[index])

            right_join_table = join_order[i]

            for join_pair in join_list:
                join_table_lst = [x[0] for x in join_pair]

                for joined_table in left_join_table_lst:
                    if joined_table in join_table_lst and right_join_table in join_table_lst:
                        multiple_join_pair.append(join_pair)

            if len(multiple_join_pair) > 1:
                ordered_join_lst.append(multiple_join_pair)
            else:
                ordered_join_lst.append(multiple_join_pair[0])
            i += 1

    return ordered_join_lst


def join_order_helper(join_list):
    join_order_helper = []
    for join_pair in join_list:
        if join_pair[0][0] not in join_order_helper:
            join_order_helper.append(join_pair[0][0])
        if join_pair[1][0] not in join_order_helper:
            join_order_helper.append(join_pair[1][0])

    return join_order_helper


# the left deep join algorithm
def left_deep_join(join_order_lst, root_path, conditions_lst, join_order_helper):
    join_result = []
    joined_table = set()
    initialize = True

    for join_pair in join_order_lst:

        if not isinstance(join_pair, list):

            left_join_table = join_pair[0][0]
            left_join_col = join_pair[0][1]
            right_join_table = join_pair[1][0]
            right_join_col = join_pair[1][1]

            if initialize:
                join_result = initial_join_table(left_join_table, right_join_table, left_join_col, right_join_col,
                                                 root_path, conditions_lst, join_order_helper)

                if len(join_result) == 0:
                    return join_result

                joined_table.add(left_join_table)
                joined_table.add(right_join_table)
                initialize = False

            else:

                if left_join_table in joined_table:

                    join_result = join_table(left_join_table, right_join_table, left_join_col, right_join_col,
                                             root_path, join_result, conditions_lst, join_order_helper)

                    if len(join_result) == 0:
                        return join_result

                    joined_table.add(right_join_table)


                elif right_join_table in joined_table:
                    join_result = join_table(right_join_table, left_join_table, right_join_col, left_join_col,
                                             root_path, join_result, conditions_lst, join_order_helper)

                    if len(join_result) == 0:
                        return join_result

                    joined_table.add(left_join_table)

        else:

            joined_table_1 = None
            joined_col_1 = None
            joined_table_2 = None
            joined_col_2 = None

            waiting_join_table = None
            waiting_join_col_1 = None
            waiting_join_col_2 = None

            for x in join_pair[0]:
                if x[0] in joined_table:
                    joined_table_1 = x[0]
                    joined_col_1 = x[1]
                else:
                    waiting_join_table = x[0]
                    waiting_join_col_1 = x[1]

            for x in join_pair[1]:
                if x[0] in joined_table:
                    joined_table_2 = x[0]
                    joined_col_2 = x[1]
                else:
                    waiting_join_table = x[0]
                    waiting_join_col_2 = x[1]

            join_result = multi_join(joined_table_1, joined_col_1, joined_table_2, joined_col_2, waiting_join_table,
                                     waiting_join_col_1, waiting_join_col_2, root_path, join_result, conditions_lst,
                                     join_order_helper)

            for i in join_order_helper:
                if i in joined_table:
                    continue
                else:
                    joined_table.add(i)
                    break

            if len(join_result) == 0:
                return join_result

    return join_result


# this function is responsible for joining 3 tables in one time which will largely lower the size of join result
def multi_join(joined_table_1, joined_col_1, joined_table_2, joined_col_2, waiting_join_table, waiting_join_col_1,
               waiting_join_col_2, root_path, join_result, conditions_list, join_order_helper):
    joined_table_1_index = join_order_helper.index(joined_table_1)

    joined_table_2_index = join_order_helper.index(joined_table_2)

    temp_join_result = []

    with open(root_path + joined_col_1 + ".dat")  as joined_file:
        joined_lst_1 = np.fromfile(joined_file, dtype="int").tolist()
    joined_file.close()

    with open(root_path + joined_col_2 + ".dat")  as joined_file:
        joined_lst_2 = np.fromfile(joined_file, dtype="int").tolist()
    joined_file.close()

    with open(root_path + waiting_join_col_1 + ".dat")  as waiting_join_file:

        waiting_join_lst_1 = np.fromfile(waiting_join_file, dtype="int").tolist()

    waiting_join_file.close()

    temp_waiting_join_lst = [[], [], []]

    for index in range(len(waiting_join_lst_1)):
        temp_waiting_join_lst[0].append(index)
        temp_waiting_join_lst[1].append(waiting_join_lst_1[index])

    waiting_join_lst = temp_waiting_join_lst

    for condition in conditions_list:
        if waiting_join_table == condition[0]:
            waiting_join_lst = filter_condition(waiting_join_table, waiting_join_col_1, waiting_join_lst,
                                                conditions_list, root_path)

            break

    with open(root_path + waiting_join_col_2 + ".dat")  as waiting_join_file:

        waiting_join_lst_2 = np.fromfile(waiting_join_file, dtype="int").tolist()

    waiting_join_file.close()

    waiting_join_lst.append([])

    for index in waiting_join_lst[0]:
        waiting_join_lst[2].append(waiting_join_lst_2[index])

    # join
    waiting_join_lst_dict_1 = dict()
    for i in range(len(waiting_join_lst[0])):
        waiting_row_value = waiting_join_lst[1][i]
        waiting_row_number = waiting_join_lst[0][i]

        if waiting_row_value in waiting_join_lst_dict_1:
            waiting_join_lst_dict_1[waiting_row_value].append(waiting_row_number)
        else:
            waiting_join_lst_dict_1[waiting_row_value] = [waiting_row_number]

    waiting_join_lst_dict_2 = dict()
    for i in range(len(waiting_join_lst[0])):
        waiting_row_value = waiting_join_lst[2][i]
        waiting_row_number = waiting_join_lst[0][i]

        if waiting_row_value in waiting_join_lst_dict_2:
            waiting_join_lst_dict_2[waiting_row_value].append(waiting_row_number)
        else:
            waiting_join_lst_dict_2[waiting_row_value] = [waiting_row_number]

    for joined_pair in join_result:
        joined_row_number_1 = joined_pair[joined_table_1_index]
        joined_row_number_2 = joined_pair[joined_table_2_index]

        joined_row_value_1 = joined_lst_1[joined_row_number_1]

        joined_row_value_2 = joined_lst_2[joined_row_number_2]

        if joined_row_value_1 in waiting_join_lst_dict_1 and joined_row_value_2 in waiting_join_lst_dict_2:
            result1 = waiting_join_lst_dict_1[joined_row_value_1]
            result2 = waiting_join_lst_dict_2[joined_row_value_2]

            valid_row_number_lst = list(set(result1) & set(result2))

            if len(valid_row_number_lst) > 0:
                for waiting_join_row_number in valid_row_number_lst:
                    temp_join_result.append(joined_pair + [waiting_join_row_number])

    join_result = temp_join_result

    return join_result



# def self_join(left_join_table, right_join_table, left_join_col, right_join_col, root_path, join_result,
#               join_order_helper):
#     temp_join_result = []
#     with open(root_path + left_join_col + ".dat")  as left_file:
#         left = np.fromfile(left_file, dtype="int").tolist()
#     left_file.close()
#
#     left_joined_table_index = int(join_order_helper.index(left_join_table))
#
#     with open(root_path + right_join_col + ".dat")  as right_file:
#         right = np.fromfile(right_file, dtype="int").tolist()
#     right_file.close()
#
#     right_joined_table_index = int(join_order_helper.index(right_join_table))
#
#     for join_pair in join_result:
#         left_index = join_pair[left_joined_table_index]
#         right_index = join_pair[right_joined_table_index]
#
#         if left[left_index] == right[right_index]:
#             temp_join_result.append(join_pair)
#
#     join_result = temp_join_result
#
#     return join_result


# This is the method responsible for joining 2 tables using hash join
def join_table(joined_table, waiting_join_table, joined_table_col, waiting_join_col, root_path, join_result,
               conditions_lst, join_order_helper):

    joined_table_index = join_order_helper.index(joined_table)

    join_result.sort(key=itemgetter(joined_table_index))

    temp_join_result = []

    with open(root_path + joined_table_col + ".dat")  as joined_file:
        joined_lst = np.fromfile(joined_file, dtype="int").tolist()
    joined_file.close()

    temp_joined_lst = [[], []]

    pointer =0

    while pointer < len(join_result):
        row_number = join_result[pointer][joined_table_index]
        if pointer == 0:
            temp_joined_lst[0].append(row_number)
            temp_joined_lst[1].append(joined_lst[row_number])
            pointer+=1

        elif row_number == join_result[pointer-1][joined_table_index]:
            pointer+=1

        elif row_number != join_result[pointer-1][joined_table_index]:
            temp_joined_lst[0].append(row_number)
            temp_joined_lst[1].append(joined_lst[row_number])
            pointer += 1


    joined_lst = temp_joined_lst




    with open(root_path + waiting_join_col + ".dat")  as waiting_join_file:

        waiting_join_lst = np.fromfile(waiting_join_file, dtype="int").tolist()

    waiting_join_file.close()
    # first inner list is row number, second inner list is value
    temp_waiting_join_lst = [[], []]

    for index in range(len(waiting_join_lst)):
        temp_waiting_join_lst[0].append(index)
        temp_waiting_join_lst[1].append(waiting_join_lst[index])
    waiting_join_lst = temp_waiting_join_lst

    for condition in conditions_lst:
        if waiting_join_table == condition[0]:
            waiting_join_lst = filter_condition(waiting_join_table, waiting_join_col, waiting_join_lst, conditions_lst,
                                                root_path)

            break

    # join part

    if len(joined_lst[0]) < len(waiting_join_lst[0]):

        joined_lst_dict = defaultdict(list)

        for i in range(len(joined_lst[0])):

            row_value = joined_lst[1][i]
            row_number = joined_lst[0][i]

            joined_lst_dict[row_value].append(row_number)


        for i in range(len(waiting_join_lst[0])):

            waiting_row_value = waiting_join_lst[1][i]
            waiting_row_number = waiting_join_lst[0][i]

            if waiting_row_value in joined_lst_dict:

                for joined_row_number in joined_lst_dict[waiting_row_value]:

                    low_join_result_index = bisect.bisect_left(KeyList(join_result, key=lambda x: x[joined_table_index]),
                                                              joined_row_number)
                    high_join_result_index = bisect.bisect_right(KeyList(join_result, key=lambda x: x[joined_table_index]),
                                                                joined_row_number)

                    for index in range(low_join_result_index, high_join_result_index):
                        temp = list(join_result[index])
                        temp.append(waiting_row_number)
                        temp_join_result.append(temp)

    else:

        waiting_join_lst_dict = defaultdict(list)

        for i in range(len(waiting_join_lst[0])):
            waiting_row_value = waiting_join_lst[1][i]
            waiting_row_number = waiting_join_lst[0][i]

            waiting_join_lst_dict[waiting_row_value].append(waiting_row_number)



        temp_result = []



        i =0
        while i < len(join_result):
            joined_row_number = join_result[i][joined_table_index]



            if i == 0:

                join_lst_index = bisect.bisect_left(joined_lst[0],
                                                    joined_row_number)

                temp_joined_row_value = joined_lst[1][join_lst_index]

                if temp_joined_row_value in waiting_join_lst_dict:
                    temp_result = waiting_join_lst_dict[temp_joined_row_value]

                    for waiting_join_row_number in temp_result:
                        temp = list(join_result[i])
                        temp.append(waiting_join_row_number)
                        temp_join_result.append(temp)
                i+=1

            elif joined_row_number != join_result[i-1][joined_table_index]:

                join_lst_index = bisect.bisect_left(joined_lst[0],
                                                    joined_row_number)

                temp_joined_row_value = joined_lst[1][join_lst_index]

                if temp_joined_row_value in waiting_join_lst_dict:
                    temp_result = waiting_join_lst_dict[temp_joined_row_value]

                    for waiting_join_row_number in temp_result:
                        temp = list(join_result[i])
                        temp.append(waiting_join_row_number)
                        temp_join_result.append(temp)
                else:
                    temp_result = []
                i+=1
            else:

                if len(temp_result)>0:
                    for waiting_join_row_number in temp_result:
                        temp = list(join_result[i])
                        temp.append(waiting_join_row_number)
                        temp_join_result.append(temp)
                i+=1


        # if joined_lst[1][0] in waiting_join_lst_dict:
        #     temp_result = waiting_join_lst_dict[joined_lst[1][0]]
        #
        #
        # left_pointer=1
        # right_pointer = 0
        #
        # while left_pointer < len(join_result) and right_pointer < len(joined_lst[0]):
        #     left_joined_row_number = join_result[left_pointer][joined_table_index]
        #     right_joined_row_number = joined_lst[0][right_pointer]
        #
        #     joined_row_value = joined_lst[1][right_pointer]
        #
        #     if left_joined_row_number != right_joined_row_number:
        #         right_pointer+=1
        #
        #     elif left_joined_row_number == right_joined_row_number and left_joined_row_number == join_result[left_pointer-1][joined_table_index]:
        #         if len(temp_result)>0:
        #             for waiting_join_row_number in temp_result:
        #                 temp = list(join_result[left_pointer])
        #                 temp.append(waiting_join_row_number)
        #                 temp_join_result.append(temp)
        #
        #         left_pointer+=1
        #     elif left_joined_row_number == right_joined_row_number and left_joined_row_number != join_result[left_pointer-1][joined_table_index]:
        #
        #         if joined_row_value in waiting_join_lst_dict:
        #             temp_result = waiting_join_lst_dict[joined_row_value]
        #             for waiting_join_row_number in temp_result:
        #                 temp = list(join_result[left_pointer])
        #                 temp.append(waiting_join_row_number)
        #                 temp_join_result.append(temp)
        #         else:
        #             temp_result=[]
        #
        #         left_pointer+=1

    join_result = temp_join_result


    return join_result



# this method is responsible for the join of first 2 tables
def initial_join_table(left_join_table, right_join_table, left_join_col, right_join_col, root_path, conditions_lst,
                       join_order_helper):
    join_result = []

    with open(root_path + left_join_col + ".dat")  as left_file:
        left = np.fromfile(left_file, dtype="int").tolist()
    left_file.close()
    temp_left = [[], []]

    for index in range(len(left)):
        temp_left[0].append(index)
        temp_left[1].append(left[index])
    left = temp_left

    for condition in conditions_lst:
        if left_join_table == condition[0]:
            left = filter_condition(left_join_table, left_join_col, left, conditions_lst, root_path)
            break

    with open(root_path + right_join_col + ".dat")  as right_file:
        right = np.fromfile(right_file, dtype="int").tolist()
    right_file.close()

    temp_right = [[], []]

    for index in range(len(right)):
        temp_right[0].append(index)
        temp_right[1].append(right[index])

    right = temp_right

    for condition in conditions_lst:
        if right_join_table == condition[0]:
            right = filter_condition(right_join_table, right_join_col, right, conditions_lst, root_path)
            break

    if len(left[0]) < len(right[0]):
        left_dict = dict()

        for i in range(len(left[0])):
            row_value = left[1][i]
            row_number = left[0][i]
            if row_value in left_dict:
                left_dict[row_value].append(row_number)
            else:
                left_dict[row_value] = [row_number]

        for i in range(len(right[0])):

            right_value = right[1][i]
            right_row_number = right[0][i]

            if right_value in left_dict:

                for left_row_number in left_dict[right_value]:
                    join_result.append([left_row_number, right_row_number])

    else:

        right_dict = dict()

        for i in range(len(right[0])):
            row_number = right[0][i]
            row_value = right[1][i]

            if row_value in right_dict:
                right_dict[row_value].append(row_number)
            else:
                right_dict[row_value] = [row_number]

        for i in range(len(left[0])):
            left_row_number = left[0][i]
            left_value = left[1][i]

            if left_value in right_dict:

                for right_row_number in right_dict[left_value]:
                    join_result.append([left_row_number, right_row_number])

    return join_result


# this method is resiposible for filtering the table according to the filter condition
def filter_condition(left_join_table, left_join_col, filter_result, conditions_list, root_path):
    for condition in conditions_list:
        condition_table = condition[0]
        condition_column = condition[1]
        condition_relation = condition[2]
        comparison_num = condition[3]

        if condition_table == left_join_table and left_join_col == condition_column:

            temp_filter_result = [[], []]

            if condition_relation == "=":
                for index in range(len(filter_result[1])):
                    if filter_result[1][index] == comparison_num:
                        temp_filter_result[0].append(filter_result[0][index])
                        temp_filter_result[1].append(filter_result[1][index])
            elif condition_relation == "<":
                for index in range(len(filter_result[1])):
                    if filter_result[1][index] < comparison_num:
                        temp_filter_result[0].append(filter_result[0][index])
                        temp_filter_result[1].append(filter_result[1][index])

            elif condition_relation == ">":
                for index in range(len(filter_result[1])):
                    if filter_result[1][index] > comparison_num:
                        temp_filter_result[0].append(filter_result[0][index])
                        temp_filter_result[1].append(filter_result[1][index])

            filter_result = temp_filter_result



        elif condition_table == left_join_table:

            temp_filter_result = [[], []]

            with open(root_path + condition_column + ".dat")  as filter_file:
                filter_lst = np.fromfile(filter_file, dtype="int").tolist()
            filter_file.close()

            if condition_relation == "=":
                for index in range(len(filter_result[0])):
                    row_number = filter_result[0][index]
                    row_value = filter_result[1][index]

                    if filter_lst[row_number] == comparison_num:
                        temp_filter_result[0].append(row_number)
                        temp_filter_result[1].append(row_value)


            elif condition_relation == "<":
                for index in range(len(filter_result[0])):
                    row_number = filter_result[0][index]
                    row_value = filter_result[1][index]

                    if filter_lst[row_number] < comparison_num:
                        temp_filter_result[0].append(row_number)
                        temp_filter_result[1].append(row_value)

            elif condition_relation == ">":
                for index in range(len(filter_result[0])):
                    row_number = filter_result[0][index]
                    row_value = filter_result[1][index]

                    if filter_lst[row_number] > comparison_num:
                        temp_filter_result[0].append(row_number)
                        temp_filter_result[1].append(row_value)

            filter_result = temp_filter_result

    return filter_result


# this method will get the sum result and return a list of that
def get_sum_result_lst(join_result, select_list, root_path, join_order_helper):
    result_list = []
    if len(join_result) == 0:
        return result_list

    for select in select_list:
        temp_sum = 0

        with open(root_path + select + ".dat")  as sum_file:
            sum_lst = np.fromfile(sum_file, dtype="int").tolist()
        sum_file.close()

        sum_index = join_order_helper.index(select[0])

        for join_pair in join_result:
            temp_sum += sum_lst[join_pair[sum_index]]

        result_list.append(temp_sum)

    return result_list

# this method is resposible for print result
def print_string(join_result, result_list, select_list):
    if len(join_result) == 0:
        for i in range(len(select_list) - 1):
            print(',', end='')
    else:

        for i in range(len(result_list)):
            print(result_list[i], end='')

            if i != len(result_list) - 1:
                print(',', end='')
    print()


# Here is the execute engine
def excutor():
    f = sys.stdin

    file_list_string = f.readline().rstrip('\n')

    num_of_sql = int(f.readline().rstrip('\n'))

    feature_dict = loader.execute_loader(file_list_string)

    root_path = loader.get_root_path(file_list_string)

    for i in range(num_of_sql):

        select_list = parser.parse_select(f.readline().rstrip('\n'))

        f.readline()

        join_list = parser.parse_join(f.readline().rstrip('\n'))

        conditions_list = parser.parse_condition(f.readline().rstrip('\n'))

        f.readline()

        temp_feature_dict = copy.deepcopy(feature_dict)

        temp_feature_dict = estimate_cardinality_after_filter(temp_feature_dict, conditions_list)

        order_helper = join_order_helper(join_list)

        rels = sorted(order_helper)

        best = selinger_join_order(feature_dict, temp_feature_dict, join_list, rels, conditions_list)

        join_order = compute_best(rels, best, temp_feature_dict, join_list, feature_dict, conditions_list)

        join_order = join_order[0]

        join_list = find_best_order_pair(join_order, join_list)

        join_result = left_deep_join(join_list, root_path, conditions_list,join_order)

        result_list = get_sum_result_lst(join_result, select_list, root_path,join_order)

        print_string(join_result, result_list, select_list)


if __name__ == "__main__":

    excutor()




    # start_time = time.time()
    # # "data/m/A.csv,data/m/B.csv,data/m/C.csv,data/m/D.csv,data/m/E.csv,data/m/F.csv,data/m/G.csv,data/m/H.csv,data/m/I.csv,data/m/J.csv,data/m/K.csv,data/m/L.csv,data/m/M.csv,data/m/N.csv,data/m/O.csv,data/m/P.csv"
    #
    # feature_dict = loader.execute_loader(
    #      "data/xs/A.csv,data/xs/B.csv,data/xs/C.csv,data/xs/D.csv,data/xs/E.csv,data/xs/F.csv")
    #
    #
    # root_path = loader.get_root_path("data/xs/A.csv,data/xs/B.csv,data/xs/C.csv,data/xs/D.csv,data/xs/E.csv,data/xs/F.csv")
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # middle_time = time.time()
    #
    # with open("data/xs/queries.sql") as sql:
    #     sql_list = sql.readlines()
    #
    # for i in range(1, len(sql_list)):
    #
    #     if i % 5 == 1:
    #         select_list = parser.parse_select(sql_list[i - 1].rstrip('\n'))
    #         print(select_list)
    #
    #
    #
    #     if i % 5 == 3:
    #         join_list = parser.parse_join(sql_list[i - 1].rstrip('\n'))
    #         print(join_list)
    #
    #
    #
    #     if i % 5 == 4:
    #         conditions_list = parser.parse_condition(sql_list[i - 1].rstrip('\n'))
    #         print(conditions_list)
    #
    #
    #
    #
    #
    #     if i > 1 and i % 5 == 0:
    #
    #         temp_feature_dict= copy.deepcopy(feature_dict)
    #
    #
    #         temp_feature_dict = estimate_cardinality_after_filter(temp_feature_dict, conditions_list)
    #
    #
    #
    #
    #
    #         order_helper=join_order_helper(join_list)
    #
    #         #best
    #
    #         rels = sorted(order_helper)
    #
    #
    #         best = selinger_join_order(feature_dict, temp_feature_dict, join_list, rels, conditions_list)
    #
    #
    #
    #
    #         join_order = compute_best(rels, best, temp_feature_dict, join_list, feature_dict, conditions_list)
    #
    #         join_order = join_order[0]
    #
    #         join_list=find_best_order_pair(join_order, join_list)
    #
    #         print(join_list)
    #         print(join_order)
    #
    #         join_result = left_deep_join(join_list, root_path, conditions_list,join_order)
    #
    #
    #         result_list = get_sum_result_lst(join_result, select_list, root_path, join_order)
    #
    #
    #         print_string(join_result, result_list, select_list)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # print("--- %s seconds ---" % (time.time() - middle_time))

