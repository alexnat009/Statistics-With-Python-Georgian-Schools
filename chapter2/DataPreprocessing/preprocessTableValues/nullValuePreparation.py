import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analysisNullValue(df, n):
	# Create bins and labels for categorizing the number of missing values
	bins = np.arange(0, n + 1, 5)
	labels = [f"{i}-{i + 5}" for i in range(0, n - 4, 5)]

	# Count the number of missing values per row
	missing_counts = df.isna().sum(axis=1)

	# Categorize rows based on the number of missing values
	categories = pd.cut(missing_counts, bins=bins, labels=labels, right=True)

	# Create a result dictionary with labels as keys and lists of tuples (school name, missing count) as values
	result = {
		label: [(df.loc[idx]['სკოლის სახელწოდება'], missing_counts[idx])
				for idx in missing_counts.index[categories == label]]
		for label in labels
	}
	return result


# noinspection SpellCheckingInspection
def removeNullValues(df):
	cols = np.array([
		'სკოლის სახელწოდება', 'შენობის მდგომარეობა',
		'საჭიროა თუ არა ახალი სკოლის აშენება',
		'სართულების რაოდენობა', 'ფასადის მდგომარეობა', 'სახურავის მდგომარეობა',
		'პანდუსის მდგომარეობა', 'გარე კარ-ფანჯრის მდგომარეობა',
		'ცენტრალური გათბობის მდგომარეობა',
		'ცენტრალური გათბობის მოწყობის ან სრული რეაბილიტაციის წელი',
		'ელექტროობის მდგომარეობა', 'წყალგაყვანილობის მდგომარეობა',
		'სველი წერტილების (საპირფარეშოების) ზოგადი მდგომარეობა',
		'საპირფარეშო ოთახების რაოდენობა', 'საკლასო ოთახების რაოდენობა',
		'I სართულის მდგომარეობა', 'II სართულის მდგომარეობა',
		'III სართულის მდგომარეობა', 'IV სართულის მდგომარეობა',
		'ადმინისტრაციის ოთახების მდგომარეობა',
		'კომპიუტერების ოთახის მდგომარეობა', 'ბიბლიოთეკის ოთახის მდგომარეობა',
		'ბუფეტის მდგომარეობა',
		'ფიზიკის/ქიმიის/ბიოლოგიის კაბინეტ-ლაბორატორიის (სამივე ლაბორატორია როცა ერთ ოთახშია) მდგომარეობა',
		'ფიზიკის კაბინეტ-ლაბორატორიის მდგომარეობა',
		'ქიმიის კაბინეტ-ლაბორატორიის მდგომარეობა',
		'ბიოლოგიის კაბინეტ-ლაბორატორიის მდგომარეობა',
		'გარე სპორტული მოედნების მდგომარეობა',
		'სპორტული დარბაზების მდგომარეობა', 'სპორტული დარბაზების რაოდენობა',
		'სააქტო დარბაზის მდგომარეობა', 'ეზოს კეთილმოწყობა', 'ფანჯრების ტიპი',
		'რით თბება სკოლის შენობა',
		'ცენტრალური გათბობა - (გაზი, დიზელი, ქვანახშირი, შეშა, ელექტროენერგია, ბრიკეტები, მზის სისტემა, სხვა)',
		'ინდივიდუალური  გათბობა (გაზზე, დიზელზე, ქვანახშირზე, შეშაზე, ელექტროენერგიაზე, ბრიკეტები, სხვა)'])
	n = 20
	missing_counts = df[cols].isna().sum()
	missing_percent = missing_counts / len(df) * 100
	percent_threshold = 40
	df.drop(columns=cols[np.in1d(cols, cols[missing_percent > percent_threshold])], inplace=True)
	cols = cols[~np.in1d(cols, cols[missing_percent > percent_threshold])]
	df.dropna(axis=0, thresh=n, inplace=True, subset=cols)
	return df


def nullValuePreprocess_main(df, output_path, save=False):
	df = removeNullValues(df)
	if save:
		df.to_pickle(output_path)
		df.to_excel(output_path.replace('.pkl', '.xlsx'), index=False)
	return df


