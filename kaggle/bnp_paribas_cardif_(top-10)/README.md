Описание проекта
----------------
Сорвенование на Kaggle, организованное одной из крупнейших страховых компаний мира BNP Paribas Cardif. Целью является предсказание одной из двух категорий претензии от клиентов.    

В качестве входного датасета имеется набор анонимизированных признаков (133) и 114 000 наблюдений.


***Подход***  
1.	Анализ корреляции признаков, отбор наиболее важных
2.	Обучаем модель Catboost и отбор важных признаков по SHAP
3.	Объединение важных признаков
4.	Генерация новых признаков на основе отобранных
5.  Финальная модель Catboost


***Результат***     
10-е место на приватном лидерборде, 13-е место на публичном