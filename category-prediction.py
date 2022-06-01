import pandas as pd

categories_tree = pd.read_csv('D:\Обучение\Data Science\Стажировка KE\Исходные файлы\categories_tree.csv')

train = pd.read_parquet('D:\Обучение\Data Science\Стажировка KE\Исходные файлы\Train.parquet', engine='pyarrow')
train.fillna(0, inplace=True)

train = train[:2000]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

tfidfconverter = CountVectorizer()
X = tfidfconverter.fit_transform(train.title).toarray()
y = train.category_id

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
