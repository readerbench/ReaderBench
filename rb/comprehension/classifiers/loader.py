import pandas as pd


def load_file(filename):
    df = pd.read_csv(filename, header=0)

    df = df.drop("name", axis=1)
    df = df.drop("class", axis=1)
    columns = df.columns
    X = []
    y = []
    for index in range(df.shape[0]):
        features = []
        if index % 2 == 0:
            this_row = df.iloc[index]
            next_row = df.iloc[index + 1]
            build_features(columns, features, next_row, this_row)
            y.append(0)
        else:
            this_row = df.iloc[index]
            prev_row = df.iloc[index - 1]
            build_features(columns, features, prev_row, this_row)
            y.append(1)
        X.append(features)
    return X, y


def build_features(columns, features, other_row, this_row):
    for column in columns:
        features.append(float(this_row[column]) / float(other_row[column]) * 100)
