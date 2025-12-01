import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/reviews.csv")
df = df.replace(['positive', 'negative'], [1, 0])
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(y_train.head())

