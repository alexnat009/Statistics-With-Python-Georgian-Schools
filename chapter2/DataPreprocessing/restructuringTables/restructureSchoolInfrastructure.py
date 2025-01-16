import re

import pandas as pd


def preprocess(df):
	# Define a list of prefixes to remove
	prefixes = ['სსიპ - ', 'სსიპ ', 'სსიპ-', 'სსიპ -', 'სსიპ', 'სსიპ- ']

	# Remove the prefixes from the beginning of each school name
	for prefix in prefixes:
		df['სკოლის სახელწოდება'] = df['სკოლის სახელწოდება'].str.replace(f'^{prefix}', '', regex=True).str.strip()

	# Add the prefix to the school name
	df['სკოლის სახელწოდება'] = 'სსიპ - ' + df['სკოლის სახელწოდება']
	df['რეგიონი'] = df['რეგიონი'].apply(lambda x: re.sub(r'\n', '', x)).str.strip()
	# Drop the column 'შენიშვნა'
	df = df.drop(columns='შენიშვნა')

	df.columns = df.columns.str.strip()

	return df



df1 = pd.read_excel('../../../Data/original/სკოლების ინფრასტრუქტურა.xlsx')
df2 = pd.read_excel('../../../Data/original/სკოლების ინფრასტრუქტურული მდგომარეობა.xlsx')
#
df1 = preprocess(df1)
df2 = preprocess(df2)

# Save the data to a new Excel file
# strip the columns
df1 = df1.rename(columns={"საიდენტიფიკაციო კოდი (ცხრანიშნა)": "კოდი (ცხრანიშნა)"})
print(df1.columns)
print(df2.columns)

df1.to_excel('../../Data/modified/modified სკოლების ინფრასტრუქტურა.xlsx', index=False)
df2.to_excel('../../Data/modified/modified სკოლების ინფრასტრუქტურული მდგომარეობა.xlsx', index=False)
