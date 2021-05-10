import pandas as pd
import noise
import os

DATA_PATH = "../../psl-datasets/movielens/data/ml-1m"
BASE_OUT_DIRECTORY =  "../../psl-datasets/movielens/data"
ATTRIBUTE_NOISE_MODELS = ["gender_flipping"]
LABEL_NOISE_MODELS = ["poisson_noise", "gaussian_noise"]
NOISE_LEVELS = {
    "poisson_noise": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "gaussian_noise": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "gender_flipping": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
}
DATA_FILES_COLS = {
    "gender_flipping": [{
        "file": "users.dat",
        "col": 1
    }],
    "poisson_noise": [{
        "file": "ratings.dat",
        "col": 2
    }],
    "gaussian_noise": [{
        "file": "users.dat",
        "col": 2
    }]
}


def generate_noisy_data(dataset_path='./', out='./out', noise_models=None, noise_levels=None, columns=None):
    if noise_models is None:
        noise_models = []
    if noise_levels is None:
        noise_levels = []
    if columns is None:
        columns = []
    dataset = pd.read_csv(dataset_path, sep='::', header=None, index_col=False)
    for index in range(len(columns)):
        col = columns[index]
        noise_method = getattr(noise, noise_models[index])
        dataset.iloc[:, int(col)] = dataset.iloc[:, int(col)].apply(lambda x: noise_method(x, noise_levels[index]))
    with open(out, 'w') as f:
        for index, row in dataset.iterrows():
            f.write('::'.join([str(elem) for elem in row]))
            f.write('\n')


def main():
    for noise_model in ATTRIBUTE_NOISE_MODELS:
        for noise_level in NOISE_LEVELS[noise_model]:
            for data_file_col in DATA_FILES_COLS[noise_model]:
                out_directory = os.path.join(BASE_OUT_DIRECTORY, "ml-1m_{}/{}".format(noise_model, noise_level))
                dataset_path = os.path.join(DATA_PATH, "{}".format(data_file_col["file"]))
                generate_noisy_data(dataset_path, out_directory, noise_model, noise_level, data_file_col["col"])


if __name__ == '__main__':
    main()
