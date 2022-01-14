import os
import sys

def choose_right_bboxes(lines, s):
    box_cnt = len(lines) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        width = x_max - x_min
        height = y_max - y_min
        if width < 2 or height < 2:  #去掉一些小的人脸
            return False
    return True

with open('data/my_data/train_ok.txt','w') as fw:    #设置文件对象
    with open("data/my_data/train.txt", "r") as fr:
        for line in fr.readlines():
            line1 = line.strip('\n').split(' ')  #去掉列表中每一个元素的换行符
            s = line.strip('\n').split(' ')
            s = s[4:]
            if len(line1) > 4 and choose_right_bboxes(line1, s):
                # print("line = ", line)
                str_line = " ".join(line1)
                fw.write(str_line)
                fw.write('\n')

# with open("./data/my_data/train_ok.txt", "r") as fr:
#     for line in fr.readlines():
#         line1 = line.strip('\n').split(' ')  #去掉列表中每一个元素的换行符
#         # s = line.strip(' ').split(' ')
#         # print("s = ", s)
#         # print("line1 = ", line1)
#         if len(line1) < 9:
#             print("********* line = ", line)