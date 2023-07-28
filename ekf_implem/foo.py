
import csv
import numpy as np
import string
vec = np.zeros((14535, 15))
with open('pred_data.csv') as file1:
    csv_reader1 = csv.reader(file1, delimiter=',')
    with open('vsc_data.csv') as file2:
        csv_reader2 = csv.reader(file2, delimiter=',')
        ln = -1
        for row in csv_reader2:
            if ln >= 0 :
                if ln ==0:
                    time_lag = float(row[10])
                vec[ln][0] = float(row[10]) - time_lag
                vec[ln][9] = float(row[8].strip('(').strip(')').strip(','))
                vec[ln][10] = float(row[9].strip('(').strip(')').strip(','))
                vec[ln][11] = float(row[7].split(',')[0].strip('('))
                vec[ln][12] = float(row[7].split(',')[1])
                vec[ln][13] = float(row[7].split(',')[2])
                vec[ln][14] = float(row[7].split(',')[3].strip(')'))
            ln+=1
            
    ln = -1
    for row in csv_reader1:
        if (ln >= 0 and ln < 14535):
            vec[ln][1] = int(row[5].split(',')[0].strip('('))
            vec[ln][2] = int(row[5].split(',')[1].strip(')'))
            vec[ln][3] = int(row[6].split(',')[0].strip('('))
            vec[ln][4] = int(row[6].split(',')[1].strip(')'))
            vec[ln][5] = int(row[7].split(',')[0].strip('('))
            vec[ln][6] = int(row[7].split(',')[1].strip(')'))
            vec[ln][7] = int(row[8].split(',')[0].strip('('))
            vec[ln][8] = int(row[8].split(',')[1].strip(')'))
        ln += 1
    


