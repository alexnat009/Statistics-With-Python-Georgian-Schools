from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('../../Data/original/მოსწ. რაოდენობა.xlsx')


def preprocess(df):
	prefixes = ['სსიპ - ', 'სსიპ ', 'სსიპ-', 'სსიპ -', 'სსიპ', 'სსიპ- ']

	# Remove the prefixes from the beginning of each school name
	for prefix in prefixes:
		df['სკოლის დასახელება'] = df['სკოლის დასახელება'].str.replace(f'^{prefix}', '', regex=True).str.strip()

	df['სკოლის დასახელება'] = 'სსიპ - ' + df['სკოლის დასახელება']

	df = df.drop(columns=['№', "საიდენტიფიკაციო კოდი", 'სკოლის ტიპი'])
	df = df.rename(columns={
		"სკოლის დასახელება": "სკოლის სახელწოდება",
		"რაიონი": "ქალაქი/მუნიციპალიტეტი",
		"საიდენტიფიკაციო კოდი.1": "კოდი (ცხრანიშნა)",
		"Code": "კოდი (ოთხნიშნა)",
		"სულ მოსწ. რაოდენობა ": "მოსწავლეთა რაოდენობა"})

	new_order = [
		'რეგიონი',
		'ქალაქი/მუნიციპალიტეტი',
		'სკოლის სახელწოდება',
		'კოდი (ოთხნიშნა)',
		'კოდი (ცხრანიშნა)',
		'მოსწავლეთა რაოდენობა'
	]
	df = df[new_order]

	df.columns = df.columns.str.strip()
	return df


# save the data to a new Excel file
df.to_excel('../../Data/modified/modified მოსწავლეთა რაოდენობა.xlsx', index=False)
