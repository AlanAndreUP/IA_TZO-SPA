import pandas as pd
from datasets import Dataset
df = pd.read_csv('data/tzotzil_spanish.csv', delimiter=',')

dataset = Dataset.from_pandas(df)
dataset.to_json('data/tzotzil_spanish_dataset.json')
