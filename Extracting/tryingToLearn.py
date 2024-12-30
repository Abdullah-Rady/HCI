from sklearn.model_selection import train_test_split
import tensorflow as tf

#tf.keras.utils.to_categorical


actions = ['rotate', 'zoomin', 'zoomout']
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(2):
        print('')