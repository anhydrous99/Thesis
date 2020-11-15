import pandas as pd


def find_max(path):
    df = pd.read_csv(path)
    return df['avg'].max()


print(find_max('../Data/data.csv'))
print(find_max('../Data/data_1.csv'))
print(find_max('../Data/data_2.csv'))
print(find_max('../Data/data_3.csv'))
print(find_max('../Data/data_4.csv'))
print(find_max('../Data/data_5.csv'))
