import pandas as pd

# with open("data/annotations.csv", encoding='utf-8') as f:
#     lines = f.readlines()
#     print(lines[6050:6060])

# Используем итератор для чтения по частям
# chunksize = 10 ** 6  # можно настроить в зависимости от RAM
# unique_values = set()

# for chunk in pd.read_csv("data/annotations.csv", chunksize=chunksize):
#     unique_values.update(chunk['text'].unique())

# print(f"Уникальные значения в поле 'text': {unique_values}")
# print(f"Количество уникальных меток: {len(unique_values)}")

# df = pd.read_csv('data/annotations.csv', sep='\t', encoding='utf-8')

# # Теперь можно обращаться по имени колонки 'text'
# unique_values = df['text'].unique()
# count_unique = df['text'].nunique()

# print("Уникальные значения:", unique_values)
# print("Количество уникальных значений:", count_unique)

# import torch
# print(torch.cuda.is_available(), torch.version.cuda)


import csv, itertools, pprint, pathlib, sys
csv_path = pathlib.Path('data/annotations.csv')
with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    print('⇢  fieldnames:', reader.fieldnames)
    first_rows = list(itertools.islice(reader, 3))
pprint.pp(first_rows)