# -*- coding: cp936 -*-
#����LabeledPoints�������ݣ���ʽ���£�
#label,factor1 factor2 .... factorn
#��һ��Ϊ����ǩ�������Կո������Ϊ���������ӣ�


import csv

reader = csv.reader(file('./data_set.csv', 'rb'))
output1 = open('LabeledPointsdata_age.txt', 'w')
output2 = open('LabeledPointsdata_sex.txt', 'w')

flag = 0
row = 0

for line in reader:
    row = row + 1
    if 1 == row:
        continue

    column = 0
    for c in line:
        column = column + 1
        if 1 == column:
            continue
        if 2 == column:
            if "��" == c:
                outputline2 = "0,"
            else:
                outputline2 = "1,"
            continue
        if 3 == column:
            outputline1 = c + ","
        else:
            if "--.--"==c:
                flag = 1
                break
            else:
                outputline1 += (c + " ")
                outputline2 += (c + " ")
    if 0 == flag:
        outputline1 += '\n'
        outputline2 += '\n'
    else:
        flag = 0
        continue
    print(column)
    output1.write(outputline1)
    output2.write(outputline2)
output1.close()
output2.close()
print('Format Successful!')