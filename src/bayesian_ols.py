import pandas as pd
from scipy.stats import norm, gamma, uniform
from tqdm import tqdm


class BayesianOLS:
    def __init__(self, X, y, param_groups, add_intercept=True) -> None:
        self.X = X.copy()
        self.add_intercept = add_intercept
        self.y = y.copy()
        self.param_groups = param_groups

        if add_intercept:
            self.X.insert(0, 'intercept', 1)

    def likelihood(self, theta):
        beta = theta.drop('sigma2')
        sigma2 = theta.loc['sigma2']
        return norm.logpdf(self.y, loc=self.X.dot(beta.T), scale=sigma2).sum()

    def prior(self, theta):
        return sum([
            i["prior"].logpdf(theta.loc[i["parameters"]]).sum() 
            for i in self.param_groups
        ])

    def posterior(self, theta):
        return self.likelihood(theta) + self.prior(theta)

    def get_init(self):
        return pd.concat([
            pd.Series(i["prior"].mean(), index=i["parameters"])
            for i in self.param_groups
        ])

    def simulate_draws(self, n_draws=1000):
        init = self.get_init()
        draws = pd.DataFrame(index=range(n_draws), columns=init.index, dtype="float")
        draws.loc[0] = init

        for i in tqdm(range(1, n_draws)):
            prev = draws.loc[i-1].copy()
            for j in self.param_groups:
                prop = prev.copy()
                prop.loc[j["parameters"]] = prev.loc[j["parameters"]] + j["proposal"].rvs()
                log_ratio = self.posterior(prop) - self.posterior(prev)
                accept = log_ratio > uniform.rvs()
                draws.loc[i, j["parameters"]] = prop.loc[j["parameters"]] if accept else prev.loc[j["parameters"]]
        self.draws = draws

    def predict(self, X):
        X = X.copy()
        beta = self.draws.drop(columns='sigma2')
        sigma2 = self.draws['sigma2']
        if self.add_intercept:
            X.insert(0, 'intercept', 1)
        pred_df = pd.DataFrame(
            data=norm.rvs(loc=X.dot(beta.T), scale=sigma2),
            index=X.index
        )
        return pred_df


def train_bayesian_ols(clean_path, features, target):
    df = pd.read_csv(clean_path, index_col='id')
    X = df.filter(features)
    y = df[target]

    param_groups = [
        {
            "parameters": ['intercept', 'vote_count', 'popularity', 'runtime'],
            "prior": norm(loc=0, scale=1),
            "proposal": norm(scale=100)
        },
        {
            "parameters": ['Horror', 'War', 'History'],
            "prior": norm(loc=0, scale=1),
            "proposal": norm(scale=100)
        },
        {
            "parameters": [
                'Dimension Films', 'Dune Entertainment', 'Screen Gems',
                'United Artists', 'Film4 Productions'
            ],
            "prior": norm(loc=0, scale=1),
            "proposal": norm(scale=100)
        },
        {
            "parameters": ['sigma2'],
            "prior": gamma(a=1, scale=3),
            "proposal": norm(scale=100)
        }
    ]

    bayesian_ols = BayesianOLS(X, y, param_groups)
    bayesian_ols.simulate_draws(n_draws=100)
    pred_df = bayesian_ols.predict(X)
    print(pred_df)
