import yaml

from src.clean_data import clean_data
from src.bayesian_ols import train_bayesian_ols

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    clean_data(config['raw_path'], config["clean_path"])
    train_bayesian_ols(config["clean_path"], config["features"], config["target"])


if __name__ == '__main__':
    main()
