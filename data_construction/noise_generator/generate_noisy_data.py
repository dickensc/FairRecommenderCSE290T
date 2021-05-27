import pandas as pd
from noise_generator import noise
import os
import shutil

DATA_PATH = "../psl-datasets/movielens/data/ml-1m"
BASE_OUT_DIRECTORY = "../psl-datasets/movielens/data"
NOISE_SETTINGS = {
    "gender_flipping": [{
        "file": "users.dat",
        "noise_level": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }],
    "label_poisson_noise": [{
        "file": "ratings.dat",
        "noise_level": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }],
    "label_gaussian_noise": [{
        "file": "ratings.dat",
        "noise_level": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }],
    "genre_gender_label_noise": [{
        "file": "ratings.dat",
        "noise_level": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        "joins": ['movies.dat', 'users.dat']
    }]
}


def get_columns(file):
    columns = None
    if file == 'users.dat':
        columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    elif file == 'movies.dat':
        columns = ['MovieID', 'Title', 'Genres']
    elif file == 'ratings.dat':
        columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    return columns


def get_mutual_column(cols1, cols2):
    return set(cols1).intersection(cols2).pop()


def load_data(directory, file, joins=None):
    if joins is None:
        joins = []
    columns = get_columns(file)
    dataset_path = os.path.join(directory, "{}".format(file))
    dataset = pd.read_csv(dataset_path, sep='::', header=None, index_col=False, engine='python', names=columns)
    for join_file in joins:
        dataset_path = os.path.join(directory, "{}".format(join_file))
        join_columns = get_columns(join_file)
        join_dataset = pd.read_csv(dataset_path, sep='::', header=None, index_col=False, engine='python', names=join_columns)
        mutual_column = get_mutual_column(columns, join_columns)
        dataset = dataset.merge(join_dataset, on=mutual_column, how='left')
    return dataset


def generate_noisy_data():
    for noise_model, noise_settings in NOISE_SETTINGS.items():
        noise_method = getattr(noise, noise_model)
        for noise_setting in noise_settings:
            joins = None
            if 'joins' in noise_setting:
                joins = noise_setting['joins']
            dataset = load_data(DATA_PATH, noise_setting["file"], joins=joins)
            for noise_level in noise_setting['noise_level']:
                out_directory = os.path.join(BASE_OUT_DIRECTORY, "ml-1m_{}/{}".format(noise_model, noise_level))
                out_path = os.path.join(out_directory, noise_setting['file'])
                edited_dataset = dataset.apply(lambda row: noise_method(row, noise_level), axis=1)
                edited_dataset = edited_dataset[get_columns(noise_setting['file'])]
                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                for file in os.listdir(DATA_PATH):
                    path = os.path.join(DATA_PATH, file)
                    out = os.path.join(out_directory, file)
                    shutil.copyfile(path, out)
                with open(out_path, 'w') as f:
                    for index, row in edited_dataset.iterrows():
                        f.write('::'.join([str(elem) for elem in row]))
                        f.write('\n')