import yaml

from src.clean_data import clean_data

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    df = clean_data(config["raw_path"])
    df.to_csv(config["clean_path"])


if __name__ == "__main__":
    main()
    