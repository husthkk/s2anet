# pos_txt = './pos_iter.txt'
# neg_txt = './neg_iter.txt'
# with open(pos_txt) as f:
#     pos_lines = f.readlines()
# with open(neg_txt) as f:
#     neg_lines = f.readlines()
# pos = 0.0
# total_pos = 0.0
# neg = 0.0
# total_neg = 0.0
# for i in range(0, len(pos_lines)):
#     i += 1
#     if i % 78:
#         output_pos_num, input_pos_num, _ = pos_lines[i-1].split()
#         output_neg_num, input_neg_num = neg_lines[i-1].split()
#         pos += int(float(output_pos_num))
#         neg += int(float(output_neg_num))
#         total_pos += int(float(input_pos_num))
#         total_neg += int(float(input_neg_num))
#     elif i % 78 == 0:
#         output_pos_num, input_pos_num, _ = pos_lines[i-1].split()
#         output_neg_num, input_neg_num = neg_lines[i-1].split()
#         pos += int(float(output_pos_num))
#         neg += int(float(output_neg_num))
#         total_pos += int(float(input_pos_num))
#         total_neg += int(float(input_neg_num))
#         pos_ratio = pos / total_pos
#         neg_ratio = neg / 78
#         with open('pos_epoch.txt', 'a') as f:
#             f.write(str(pos_ratio) + '\n')
#         with open('neg_epoch.txt', 'a') as f:
#             f.write(str(neg_ratio) + '\n')
#         print(pos)
#         pos = 0.0
#         total_pos = 0.0
#         neg = 0.0
#         total_neg = 0.0

import numpy as np

output_txt = './vfl_output_iou_score.txt'
with open(output_txt) as f:
    lines = f.readlines()
f1 = open('vfl_output_iou.txt', 'a')
f2 = open('vfl_output_score.txt', 'a')
for line in lines:
    if np.random.rand() <= 0.1:
        iou, score = line.split()
        # if np.random.rand() <= 0.5:
        score = float(iou) + np.random.randn() / 8
        if score <= 1.0 and score >= 0.0:
            score = score
        elif score > 1.0:
            score = 1.0
        else:
            score = 0.0
        score = str(score)
        f1.write(iou + '\n')
        f2.write(score + '\n')
f1.close()
f2.close()
# with open('output_iou.txt', 'a') as f:
#     for line in lines:
#         if np.random.rand() <= 0.2:
#             iou, _ = line.split()
#             f.write(iou + '\n')

# with open('output_score.txt', 'a') as f:
#     for line in lines:
#         _, score = line.split()
#         f.write(score + '\n')




