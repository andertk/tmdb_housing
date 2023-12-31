import yaml
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.clean_data import clean_data
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from numpy import log10

from src.evaluate_model import ModelEvaluator

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

df = pd.read_csv(config['clean_path'], index_col='id')
y = df[config['target']]
X = df[config['features']]

pipe = [('pt', PowerTransformer()), ('ols', LinearRegression())]
model = Pipeline(pipe)
model.fit(X, y)

yhat = pd.Series(model.predict(X), index=y.index, name='yhat')

p = ModelEvaluator(X.filter(['vote_count', 'popularity']).apply(log10), y, yhat)
p.x_y_grid()
p.yhat_y_scatter()
p.res_kde()
p.yhat_res_scatter(alpha=0.5, s=15)


r2_score(y, yhat)
model['ols'].coef_