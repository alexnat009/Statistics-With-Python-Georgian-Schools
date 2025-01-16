import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

from chapter2.regressionImplementation.utils.util_functions import find_optimal_alpha

withDifferentBuildings = True
name = 'withDifferentBuildings' if withDifferentBuildings else 'withoutDifferentBuildings'
df = pd.read_pickle(f'../../Data/modified/finalDatabase/{name}/finalDatabase_{name}.pkl')

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

tmp = df[cols].select_dtypes('number').corr()["2023 წლის ბიუჯეტი"].sort_values(ascending=False).nlargest(5)
working_df = df[tmp.index].dropna().copy()

X = working_df[tmp.index].drop(columns=["2023 წლის ბიუჯეტი"])
Y = working_df["2023 წლის ბიუჯეტი"]

X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, random_state=42, test_size=0.2)

region_div = [(region_name, group) for region_name, group in df.groupby('რეგიონი')]

X_div = []
Y_div = []
for (_, region) in region_div:

	working_cols = region[cols].select_dtypes('number').corr()["2023 წლის ბიუჯეტი"].sort_values(
		ascending=False).nlargest(
		5)
	working_df = region[working_cols.index].dropna().copy()
	if "2023 წლის ბიუჯეტი" in working_df.columns:
		X_div.append(working_df[working_cols.index].drop(columns=["2023 წლის ბიუჯეტი"]))
		Y_div.append(working_df["2023 წლის ბიუჯეტი"])


X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, random_state=42, test_size=0.15)
alpha, _ = find_optimal_alpha(np.logspace(0, 7, num=1000), Ridge(), X_train, y_train, X_test, y_test)
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)

scores = np.array([ridge.score(x.values, y.values) for x, y in zip(X_div, Y_div)])


for score, (region_name, _) in zip(scores, region_div):
	print(f"Test score mean: {score} for the region {region_name}")
