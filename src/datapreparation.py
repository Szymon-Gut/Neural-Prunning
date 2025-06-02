import pickle
import numpy as np
import random

def load_cifar_batch(filepath):
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    return data_dict

train_data = load_cifar_batch("./dataset/train")
test_data = load_cifar_batch("./dataset/test")
meta_data = load_cifar_batch("./dataset/meta")

fine_label_names = meta_data["fine_label_names"]
coarse_label_names = meta_data["coarse_label_names"]

people_classes = ["baby", "boy", "girl", "man", "woman"]
household_electrical_devices_classes = ["clock", "telephone", "television", "keyboard", "lamp"]
other_classes = ["bed", "chair", "couch", "table", "bowl", "cup", "plate", "wardrobe"]
all_selected_classes = people_classes + household_electrical_devices_classes + other_classes

selected_class_indices = {name: fine_label_names.index(name) for name in all_selected_classes}

label_map = {}
for name in people_classes:
    idx = fine_label_names.index(name)
    label_map[idx] = "people"
for name in household_electrical_devices_classes:
    idx = fine_label_names.index(name)
    label_map[idx] = "household_electrical_devices"
for name in other_classes:
    idx = fine_label_names.index(name)
    label_map[idx] = name

def filter_and_remap(X, y, label_map):
    filtered_X = []
    filtered_y = []
    for img, label_idx in zip(X, y):
        if label_idx in label_map:
            filtered_X.append(img)
            filtered_y.append(label_map[label_idx])
    return np.array(filtered_X), np.array(filtered_y)

X_train, y_train = train_data["data"], train_data["fine_labels"]
X_test, y_test = test_data["data"], test_data["fine_labels"]

filtered_X_train, filtered_y_train = filter_and_remap(X_train, y_train, label_map)
filtered_X_test, filtered_y_test = filter_and_remap(X_test, y_test, label_map)

np.random.seed(42)
random_percentage = 0.1
n_random = int(len(filtered_y_train) * random_percentage)
random_indices = np.random.choice(len(filtered_y_train), n_random, replace=False)

unique_labels = list(set(filtered_y_train))

for idx in random_indices:
    original_label = filtered_y_train[idx]
    possible_wrong_labels = [l for l in unique_labels if l != original_label]
    filtered_y_train[idx] = random.choice(possible_wrong_labels)

with open("./dataset/filtered_dataset.pkl", "wb") as f:
    pickle.dump({
        "X_train": filtered_X_train,
        "y_train": filtered_y_train,
        "X_test": filtered_X_test,
        "y_test": filtered_y_test
    }, f)