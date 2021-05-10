import pandas as pd
from noise_generator import noise
import os

DATA_PATH = "../psl-datasets/movielens/data/ml-1m"
BASE_OUT_DIRECTORY = "../psl-datasets/movielens/data"
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
        "col": [1]
    }],
    "poisson_noise": [{
        "file": "ratings.dat",
        "col": [2]
    }],
    "gaussian_noise": [{
        "file": "ratings.dat",
        "col": [2]
    }]
}


def generate_noisy_data():
    for noise_model in ATTRIBUTE_NOISE_MODELS:
        for noise_level in NOISE_LEVELS[noise_model]:
            for data_file_col in DATA_FILES_COLS[noise_model]:
                out_directory = os.path.join(BASE_OUT_DIRECTORY, "ml-1m_{}/{}".format(noise_model, noise_level))
                out_path = os.path.join(out_directory, data_file_col["file"])
                dataset_path = os.path.join(DATA_PATH, "{}".format(data_file_col["file"]))
                dataset = pd.read_csv(dataset_path, sep='::', header=None, index_col=False, engine='python')
                for index in range(len(data_file_col["col"])):
                    col = data_file_col["col"][index]
                    noise_method = getattr(noise, noise_model)
                    dataset.iloc[:, int(col)] = dataset.iloc[:, int(col)].apply(lambda x: noise_method(x, noise_level))
                # Create data directory to write output to
                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                with open(out_path, 'w') as f:
                    for index, row in dataset.iterrows():
                        f.write('::'.join([str(elem) for elem in row]))
                        f.write('\n')

    for noise_model in LABEL_NOISE_MODELS:
        for noise_level in NOISE_LEVELS[noise_model]:
            for data_file_col in DATA_FILES_COLS[noise_model]:
                out_directory = os.path.join(BASE_OUT_DIRECTORY, "ml-1m_{}/{}".format(noise_model, noise_level))
                out_path = os.path.join(out_directory, data_file_col["file"])
                dataset_path = os.path.join(DATA_PATH, "{}".format(data_file_col["file"]))
                dataset = pd.read_csv(dataset_path, sep='::', header=None, index_col=False, engine='python')
                for index in range(len(data_file_col["col"])):
                    col = data_file_col["col"][index]
                    noise_method = getattr(noise, noise_model)
                    dataset.iloc[:, int(col)] = dataset.iloc[:, int(col)].apply(lambda x: noise_method(x, noise_level))
                # Create data directory to write output to
                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                with open(out_path, 'w') as f:
                    for index, row in dataset.iterrows():
                        f.write('::'.join([str(elem) for elem in row]))
                        f.write('\n')
