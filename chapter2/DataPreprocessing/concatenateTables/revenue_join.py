import pandas as pd


def join4_revenue(df1, df2, withDifferentBuildings, save=False, outputPath=None):
	if withDifferentBuildings:
		df_combined = pd.merge(df1, df2, on='კოდი (ცხრანიშნა)')
		df_combined = df_combined.drop(columns=['რეგიონი_y', 'ქალაქი/მუნიციპალიტეტი_y', 'სკოლის სახელწოდება_y', ])
		df_combined = df_combined.rename(columns={'რეგიონი_x': 'რეგიონი',
												  'ქალაქი/მუნიციპალიტეტი_x': 'ქალაქი/მუნიციპალიტეტი',
												  'სკოლის სახელწოდება_x': 'სკოლის სახელწოდება',
												  })
	else:
		df2 = df2.rename(columns={'სკოლის სახელწოდება': 'სკოლის სახელწოდება.1'})

		df1.set_index(['რეგიონი', 'ქალაქი/მუნიციპალიტეტი', 'კოდი (ცხრანიშნა)'], inplace=True)
		df2.set_index(['რეგიონი', 'ქალაქი/მუნიციპალიტეტი', 'კოდი (ცხრანიშნა)'], inplace=True)

		df_combined = pd.concat([df1, df2], axis=1, join='inner').reset_index()

		df_combined = df_combined.drop(columns=['სკოლის სახელწოდება.1'])

		column_to_move = 'კოდი (ცხრანიშნა)'
		n = 4

		column = df_combined.pop(column_to_move)
		df_combined.insert(n, column_to_move, column)
	if save:
		df_combined.to_excel(outputPath, index=False)
	return df_combined
