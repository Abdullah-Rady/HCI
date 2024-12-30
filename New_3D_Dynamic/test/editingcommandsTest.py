import time
import random

actions = ['rotatec', 'rotateac', 'zoomin', 'zoomout']
y = 0
for x in range(600):
    y += 1
    if y > 3:
        y = 0
    with open('commands.txt', 'w') as f:
        # f.write(actions[random.randint(0, 3)] + ':0.1')
        f.write(actions[y] + ':0.1')

    # time.sleep(1)
