from sklearn.datasets import load_iris
import pandas as pd

raw_data = load_iris()

print raw_data.feature_names
print len(raw_data.target)
print len(raw_data.data)