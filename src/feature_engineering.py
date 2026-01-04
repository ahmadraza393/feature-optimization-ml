from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df, columns, method='standard'):
    """Scale features using StandardScaler or MinMaxScaler"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method")
    df[columns] = scaler.fit_transform(df[columns])
    return df
