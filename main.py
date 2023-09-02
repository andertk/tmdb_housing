from src.clean_data import clean_data
from src.bayesian_ols import train_bayesian_ols


def main():
    clean_data()
    train_bayesian_ols()


if __name__ == '__main__':
    main()
