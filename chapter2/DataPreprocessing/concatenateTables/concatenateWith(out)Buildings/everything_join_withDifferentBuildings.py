import pandas as pd

inputFile1 = '../../../Data/modified/joins/infrastructure_condition_studentNumber_counts_join_withDifferentBuildings.xlsx'
inputFile2 = '../../../Data/modified/modified სკოლების შემოსავლები და გასავლები.xlsx'
outputFile = ('../../../Data/modified/joins'
			  '/infrastructure_condition_studentNumber_counts_revenue_join_withDifferentBuildings.xlsx')

df1 = pd.read_excel(inputFile1)
df2 = pd.read_excel(inputFile2)

df_combined = pd.merge(df1, df2, on='კოდი (ცხრანიშნა)')
df_combined = df_combined.drop(columns=['რეგიონი_y', 'ქალაქი/მუნიციპალიტეტი_y', 'სკოლის სახელწოდება_y', ])
df_combined = df_combined.rename(columns={'რეგიონი_x': 'რეგიონი',
										  'ქალაქი/მუნიციპალიტეტი_x': 'ქალაქი/მუნიციპალიტეტი',
										  'სკოლის სახელწოდება_x': 'სკოლის სახელწოდება',
										  })

# df_combined.to_excel(outputFile, index=False)
