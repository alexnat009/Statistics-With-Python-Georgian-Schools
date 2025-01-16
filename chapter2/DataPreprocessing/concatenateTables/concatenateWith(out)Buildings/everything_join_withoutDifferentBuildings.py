import pandas as pd

inputFile1 = '../../Data/modified/joins/infrastructure_condition_studentNumber_counts_join_withoutDifferentBuildings.xlsx'
inputFile2 = '../../Data/modified/modified სკოლების შემოსავლები და გასავლები.xlsx'
outputFile = ('../../Data/modified/joins'
			  '/infrastructure_condition_studentNumber_counts_revenue_join_withoutDifferentBuildings.xlsx')

df1 = pd.read_excel(inputFile1)
df2 = pd.read_excel(inputFile2)

df2 = df2.rename(columns={'სკოლის სახელწოდება': 'სკოლის სახელწოდება.1'})

df1.set_index(['რეგიონი', 'ქალაქი/მუნიციპალიტეტი', 'კოდი (ცხრანიშნა)'], inplace=True)
df2.set_index(['რეგიონი', 'ქალაქი/მუნიციპალიტეტი', 'კოდი (ცხრანიშნა)'], inplace=True)

df_combined = pd.concat([df1, df2], axis=1, join='inner').reset_index()

df_combined = df_combined.drop(columns=['სკოლის სახელწოდება.1'])

column_to_move = 'კოდი (ცხრანიშნა)'
n = 4

column = df_combined.pop(column_to_move)
df_combined.insert(n, column_to_move, column)

df_combined.to_excel(outputFile, index=False)
