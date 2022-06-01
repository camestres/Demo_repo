import pandas as pd

# Для импорта входных данных в формате *.parquet в Pandas был использован метод read_parquet.
categories_tree = pd.read_csv('D:\Обучение\Data Science\Стажировка KE\Исходные файлы\categories_tree.csv')

train = pd.read_parquet('D:\Обучение\Data Science\Стажировка KE\Исходные файлы\Train.parquet', engine='pyarrow')
train.fillna(0, inplace=True)

# К сожлению, ввиду ограниченной производительности имеющейся техники приходится ограничивать датасет для
# возможности демонстрации логики решения. При этом есть четкое понимание, что объем данных можно сократить
# для повышения производительности, но для этого нужно больше времени, чтобы разобраться, учитывая мой текущий уровень.
train = train[:2000]

#В качестве метода преобразования текста выбран CountVectorizer, как один из наиболее
#часто применяющихся, учитывающий частоту повторения слов
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

tfidfconverter = CountVectorizer()
X = tfidfconverter.fit_transform(train.title).toarray()
y = train.category_id

#На основании данных, предоставленных в файле “train.parquet” была проведена разбивка датафрейма на обучающую и тестовую методом train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#В качестве метода обучения модели выбран RandomForestClassifier, учитывая специфику
#задачи по распознаванию текста.
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#Для оценки результативности модели был рассчитан классический коэффициент F1-score
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))