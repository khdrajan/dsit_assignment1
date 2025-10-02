import pandas as pd, yaml, sys
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))
raw = params['data']['raw_csv']
out = params['data']['processed_csv']

if __name__ == '__main__':
    df = pd.read_csv(raw)
    # Basic cleaning: drop exact dupes, strip spaces, normalize text columns
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    # Save
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Wrote {out} with {len(df):,} rows')
