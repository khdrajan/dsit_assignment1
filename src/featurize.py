import pandas as pd, yaml
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))
proc = params['data']['processed_csv']

if __name__ == '__main__':
    df = pd.read_csv(proc)
    # Example engineered features
    df['cost_per_person'] = (df['cost'] / 2).fillna(df['cost'].median()/2)
    df['has_groupon'] = df.get('groupon', pd.Series(False, index=df.index)).astype(int)
    # Cuisine diversity proxy = count of commas + 1
    df['cuisine_count'] = df['cuisine'].fillna('').str.count(',') + (df['cuisine'].notna()).astype(int)
    df.to_csv(proc, index=False)
    print('Featurization complete -> overwrote processed CSV')
