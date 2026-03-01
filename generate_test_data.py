import pandas as pd
import numpy as np

def create_messy_data(filename="messy_data.csv"):
    rng = np.random.default_rng(42)
    n_rows = 100
    
    # Create valid data
    data = {
        'ID': range(1, n_rows + 1),
        'Name': [f"User_{i}" for i in range(n_rows)],
        'Joined_Date': pd.date_range(start='2020-01-01', periods=n_rows, freq='D').strftime('%Y-%m-%d'),
        'Age': rng.integers(18, 80, size=n_rows).astype(float),
        'Income': rng.normal(50000, 15000, size=n_rows),
        'Score': rng.uniform(0, 100, size=n_rows),
        'Empty_Col_1': [np.nan] * n_rows, # Entirely empty column
        'Empty_Col_2': [None] * n_rows,  # Entirely empty column
        'Category': rng.choice(['A', 'B', 'C', None], size=n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce NaNs in Age and Income
    df.loc[rng.choice(df.index, size=10, replace=False), 'Age'] = np.nan
    df.loc[rng.choice(df.index, size=15, replace=False), 'Income'] = np.nan
    
    # Introduce Outliers
    df.loc[0, 'Income'] = 1000000  # Massive outlier
    df.loc[1, 'Age'] = 250         # Impossible age
    df.loc[2, 'Score'] = -500      # Impossible score
    
    # Introduce Duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    df.to_csv(filename, index=False)
    print(f"Created messy dataset: {filename}")

if __name__ == "__main__":
    create_messy_data()
