import pandas as pd, yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))
proc = params['data']['processed_csv']

if __name__ == '__main__':
    df = pd.read_csv(proc)
    # Binary label from rating_text
    good = ['Good','Very Good','Excellent']
    df = df.dropna(subset=['rating_text'])
    df['label'] = df['rating_text'].apply(lambda x: 1 if str(x) in good else 0)
    num_cols = ['cost','votes','cost_per_person','cuisine_count']
    cat_cols = ['type','subzone']
    X = df[num_cols + cat_cols]
    y = df['label']
    pre = ColumnTransformer([
        ('num','passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    pipe = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['model']['test_size'], random_state=params['seed'], stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rep = classification_report(y_test, y_pred)
    Path('models').mkdir(exist_ok=True)
    joblib.dump(pipe, 'models/logreg.joblib')
    with open('reports/logreg_report.txt','w') as f:
        f.write(rep)
    print(rep)
