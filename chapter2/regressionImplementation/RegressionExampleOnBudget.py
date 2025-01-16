import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

from chapter2.regressionImplementation.utils.util_functions import find_optimal_alpha

withDifferentBuildings = False
name = 'withDifferentBuildings' if withDifferentBuildings else 'withoutDifferentBuildings'
df = pd.read_pickle(f'../../Data/modified/finalDatabase/{name}/finalDatabase_{name}.pkl')

# print(df.dtypes)
for col in df.columns:
	print(df[col].dtype)
categorical_cols = df.select_dtypes(include=['category']).columns

df[categorical_cols] = df[categorical_cols].apply(lambda x: x.cat.codes.where(x.cat.codes >= 0, np.nan))

cols = ['რეგიონი', 'ქალაქი/მუნიციპალიტეტი', 'სკოლის სახელწოდება', 'კოდი (ცხრანიშნა)', 'კოდი (ოთხნიშნა)',
		'მოსწავლეთა რაოდენობა', 'შენობის მდგომარეობა',
		'საჭიროა თუ არა ახალი სკოლის აშენება',
		'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)', 'ეზოს ფართობი (კვ.მ)',
		'სართულების რაოდენობა', 'ფასადის მდგომარეობა', 'სახურავის მდგომარეობა',
		'პანდუსის მდგომარეობა', 'გარე კარ-ფანჯრის მდგომარეობა',
		'ცენტრალური გათბობის მდგომარეობა', 'ელექტროობის მდგომარეობა',
		'წყალგაყვანილობის მდგომარეობა',
		'სველი წერტილების (საპირფარეშოების) ზოგადი მდგომარეობა',
		'საპირფარეშო ოთახების რაოდენობა', 'საკლასო ოთახების რაოდენობა',
		'I სართულის მდგომარეობა', 'II სართულის მდგომარეობა',
		'ადმინისტრაციის ოთახების მდგომარეობა',
		'კომპიუტერების ოთახის მდგომარეობა', 'ბიბლიოთეკის ოთახის მდგომარეობა',
		'ბუფეტის მდგომარეობა', 'გარე სპორტული მოედნების მდგომარეობა',
		'სპორტული დარბაზების მდგომარეობა', 'სპორტული დარბაზების რაოდენობა',
		'სააქტო დარბაზის მდგომარეობა', 'ეზოს კეთილმოწყობა', 'ფანჯრების ტიპი',
		'რით თბება სკოლის შენობა',
		'ცენტრალური გათბობა - (გაზი, დიზელი, ქვანახშირი, შეშა, ელექტროენერგია, ბრიკეტები, მზის სისტემა, სხვა)',
		'შენობების რაოდენობა', 'ცვლების რაოდენობა', '2023 წლის ბიუჯეტი']

working_cols = df[cols].select_dtypes('number').corr()["2023 წლის ბიუჯეტი"].sort_values(ascending=False).nlargest(2)
print(working_cols)
working_df = df[working_cols.index].dropna().copy()

X = working_df.drop(columns=["2023 წლის ბიუჯეტი"])
Y = working_df["2023 წლის ბიუჯეტი"]

X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, random_state=42, test_size=0.2)

lr = LinearRegression()

lr.fit(X_train, y_train)
print("Linear Regression")
print("Train score:", lr.score(X_train, y_train))
print("Test score:", lr.score(X_test, y_test))

alpha, _ = find_optimal_alpha(np.logspace(0, 7, num=1000), Ridge(), X_train, y_train, X_test, y_test, plot=True,
							  save=False, filepath='../../chapter2/graphs/ridge_alpha.png')
print(f"Optimal alpha: {alpha}")
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
print("Ridge Regression")
print("Train score:", ridge.score(X_train, y_train))
print("Test score:", ridge.score(X_test, y_test))
print()

# predict some concrete value
school = {
	'მოსწავლეთა რაოდენობა': 300,
	# 'საკლასო ოთახების რაოდენობა': 15,
	# 'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)': 2740,
	# 'სართულების რაოდენობა': 2,
}
print(f'Predicting budget for school with: {json.dumps(school, sort_keys=True, indent=4, ensure_ascii=False)}')
print(ridge.predict([list(school.values())]))
print(lr.predict([list(school.values())]))
