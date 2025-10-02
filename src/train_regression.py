import pandas as pd, yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))
proc = params['data']['processed_csv']

if __name__ == '__main__':
    df = pd.read_csv(proc)
    df = df.dropna(subset=['rating_number'])
    num_cols = ['cost', 'votes', 'cost_per_person', 'cuisine_count']
    X = df[num_cols].fillna(df[num_cols].median())
    y = df['rating_number']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['model']['test_size'], random_state=params['seed'])
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    Path('models').mkdir(exist_ok=True)
    joblib.dump(lr, 'models/linreg.joblib')
    with open('reports/linreg_mse.txt','w') as f:
        f.write(str(mse))
    print('MSE:', mse)
