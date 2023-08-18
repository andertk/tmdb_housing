import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix


class ModelEvaluator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, yhat: pd.Series):
        self.X = X
        self.y = y
        self.yhat = yhat
        self.res = pd.Series(y - yhat, name="residual")

    def yhat_y_scatter(self, **kwargs):
        plt.scatter(self.yhat, self.y, **kwargs)
        plt.xlabel(self.yhat.name)
        plt.ylabel(self.y.name)
        plt.show()

    def yhat_res_scatter(self, **kwargs):
        plt.scatter(self.y, self.yhat, **kwargs)
        plt.xlabel(self.y.name)
        plt.ylabel("Residuals")
        plt.show()

    def res_kde(self, **kwargs):
        self.res.plot.kde(**kwargs, title="Residuals")
        plt.show()

    def scatter_plot_res(self, **kwargs):
        scatter_matrix(self.X.join(self.res), **kwargs)
        plt.show()
