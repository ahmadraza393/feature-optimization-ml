import pandas as pd

def load_data(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def save_processed_data(df, path):
    """Save processed dataset"""
    df.to_csv(path, index=False)
