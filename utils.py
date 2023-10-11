import os
import pickle

def make_dir(path):
    num = 1
    while f"model{num}" in os.listdir(path):
        num += 1
    path = f'{path}/model{num}'
    os.mkdir(path)
    return path

def save_var(path, name, var):
    path = path + name
    with open(f'{path}.p', 'wb') as f:
        pickle.dump(var, f)