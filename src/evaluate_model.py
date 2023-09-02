import matplotlib.pyplot as plt
import pandas as pd


def evaluate_convergence(draws):
    acceptance_rate = (draws.diff() != 0).mean()
    print(acceptance_rate)

    corr_df = draws.corr()
    print(corr_df)

    n_cols = 4
    draws.plot(
        subplots=True, 
        layout=(draws.shape[1] // n_cols + 1, n_cols),
        figsize=(16, 8)
    )
    plt.show()


class ModelEvaluator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, yhat: pd.Series):
        self.df = X.join(y).join(yhat).assign(residual=y-yhat)
        self.x_names = X.columns.to_list()
        self.y_name = y.name
        self.yhat_name = yhat.name

    def yhat_y_scatter(self, **kwargs):
        ax = self.df.plot.scatter(self.yhat_name, self.y_name, **kwargs)
        ax.axline(xy1=(0, 0), slope=1, c='black')
        plt.show()

    def yhat_res_scatter(self, **kwargs):
        self.df.plot.scatter(self.yhat_name, 'residual', **kwargs)
        plt.show()

    def res_kde(self, **kwargs):
        self.df['residual'].plot.kde(**kwargs, title='Residuals')
        plt.show()

    def x_y_grid(self, x_cols=None, scatter_kws=None, line_kws=None):
        if x_cols is None:
            x_cols = self.x_names

        if scatter_kws is None:
            scatter_kws = {}

        if line_kws is None:
            line_kws = {'c': 'black'}
        
        for i in x_cols:
            df_i = self.df.sort_values(i)
            ax = df_i.plot.scatter(i, self.y_name, **scatter_kws)
            df_i.plot.line(i, self.yhat_name, ax=ax, **line_kws)
            plt.show()
