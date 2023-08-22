import matplotlib.pyplot as plt
import pandas as pd


class ModelEvaluator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, yhat: pd.Series):
        self.df = X.join(y).join(yhat).assign(residual=y-yhat)
        self.x_names = X.columns.to_list()
        self.y_name = y.name
        self.yhat_name = yhat.name

    def yhat_y_scatter(self, **kwargs):
        self.df.plot.scatter(self.yhat_name, self.y_name, **kwargs)
        plt.show()

    def yhat_res_scatter(self, **kwargs):
        self.df.plot.scatter(self.yhat_name, "residual", **kwargs)
        plt.show()

    def res_kde(self, **kwargs):
        self.df["residual"].plot.kde(**kwargs, title="Residuals")
        plt.show()

    def x_y_grid(self, x_cols=None, scatter_kws=None, line_kws=None):
        if x_cols is None:
            x_cols = self.x_names

        if scatter_kws is None:
            scatter_kws = {}

        if line_kws is None:
            line_kws = {}
        
        for i in x_cols:
            df_i = self.df.sort_values(i)
            ax = df_i.plot.scatter(i, self.y_name, **scatter_kws)
            df_i.plot.line(i, self.yhat_name, c="black", ax=ax, **line_kws)
            plt.show()
