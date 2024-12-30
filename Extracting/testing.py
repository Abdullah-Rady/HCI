# import os
# import numpy as np
#
# print(os.listdir('Old/ZoomIn'))
# bad = []
# for i in range(1, 20):
#     # a = np.loadtxt('Rot/rotClockWiseNew_Omar_'+str(i))
#     # a = np.loadtxt('ZIn/ZoomInNew_Omar_'+str(i))
#     a = np.loadtxt('ZOut/ZoomOutNew_Omar_'+str(i))
#
#     c = 0
#     for i in a:
#         for j in i:
#             c += 1
#     bad.append(c)
# print(bad)
# for x in bad:
#     if x % 63 != 0:
#         print(bad.index(x))
#
#
#
# # Actions that we try to detect
# actions = np.array(['rotatec', 'zoomin', 'zoomout'])
#
# # # Thirty videos worth of data
# # no_sequences = 17
# #
# # # Videos are going to be 30 frames in length
# # sequence_length = 12
#
# label_map = {label:num for num, label in enumerate(actions)}
# sequences, labels = [], []
# for action in actions:
#     print(action)
# print(label_map[actions[0]])
# for sequence in range(1, 20):
#     window = []
#     a = np.loadtxt('Rot/rotClockWiseNew_Omar_'+str(sequence))
#     for frame_num in range(len(a)):
#         res = a[frame_num]
#         window.append(res)
#     sequences.append(window)
#     labels.append(label_map[actions[0]])
#
# print(np.array(sequences, dtype=object).shape)
# print(np.array(labels, dtype=object).shape)

for i in range(1, 20):
    print(i)