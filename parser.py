def parse_select(line):
    select = []
    for i in line[6:].split(","):
        select.append(i[5: len(i)-1])
    return select


def parse_join(line):
    join = []
    for pair in line[5:].split("AND"):
        left = pair.split("=")[0][1:5]
        right = pair.split("=")[1][1:5]

        join.append((get_table_column_tuple(left), get_table_column_tuple(right)))

    return join


def parse_condition(line):

    conditons = []
    line = line[4: len(line)-1]
    condition_list = line.split(" AND ")

    for condition in condition_list:

        condition_elements = condition.split(" ")
        table_name = condition_elements[0][0]
        col_name = condition_elements[0]
        relation = condition_elements[1]
        number  = int(condition_elements[2])
        conditons.append((table_name, col_name, relation, number))
    return conditons




def get_table_column_tuple(table_col):
    table = table_col[0]
    col = table_col
    return (table, col)




if __name__ == "__main__":
    pass

