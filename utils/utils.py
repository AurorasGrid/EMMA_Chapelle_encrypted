import numpy as np
import csv


def downsample(vector, factor):
    return vector[::int(factor)]


def load_csv_1vector(csv_path):
    vector = load_csv_1row(csv_path)
    if vector.shape[0] == 1:
        vector = load_csv_1col(csv_path)
    return vector


def load_csv_1row(csv_path):
    try:
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            row = np.asarray([float(i) for i in next(reader)])
        csvFile.close()
        return row
    except Exception as e:
        print(e)
        return np.zeros(1)


def load_csv_1col(csv_path):
    try:
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            col = []
            for row in reader:
                col.append(float(row[0]))
        csvFile.close()
        return np.array(col)
    except Exception as e:
        print(e)
        np.zeros(1)


def load_csv_mult_row(csv_path):
    try:
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            rows = []
            for row in reader:
                rows.append([float(i) for i in row])
        csvFile.close()
        return np.array(rows)
    except Exception as e:
        print(e)
        np.zeros(1)


def append_to_file(filename, text):
    try:
        with open(filename, "a") as myfile:
            myfile.write(text)
    except:
        print(' - Error : appending to file failed')


def read_exit_flag(exitFlagName):
    f = open(exitFlagName, "r")
    flag = f.read()
    if flag == "0":
        return False
    else:
        return True


def write_exit_flag(exitFlagName, flag):
    f = open(exitFlagName, "w+")
    if flag == False:
        f.write("0")
    else:
        f.write("1")
    f.close()

