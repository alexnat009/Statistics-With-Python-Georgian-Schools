import pandas as pd
from chapter2.DataPreprocessing.concatenateTables.infrastructure_condition_join import join1_infrastructure_condition
from chapter2.DataPreprocessing.concatenateTables.infrastructure_condition_stundetNumber_join import join2_studentNumber
from chapter2.DataPreprocessing.concatenateTables.infrastructure_condition_studentNumber_building_join import \
	join3_school_building
from chapter2.DataPreprocessing.concatenateTables.revenue_join import join4_revenue

inputFile1 = '../../../../Data/modified/modified სკოლების ინფრასტრუქტურა.xlsx'
inputFile2 = '../../../../Data/modified/modified სკოლების ინფრასტრუქტურული მდგომარეობა.xlsx'
inputFile3 = '../../../../Data/modified/modified მოსწავლეთა რაოდენობა.xlsx'
inputFile4 = '../../../../Data/modified/modified სკოლების შემოსავლები და გასავლები.xlsx'
df1 = pd.read_excel(inputFile1)
df2 = pd.read_excel(inputFile2)
df3 = pd.read_excel(inputFile3)
df4 = pd.read_excel(inputFile4)

withoutDifferentBuildings = False
withDifferentBuildings = not withoutDifferentBuildings
name = 'withDifferentBuildings' if withDifferentBuildings else 'withoutDifferentBuildings'

outputFile_join1 = f'../../../../Data/modified/joins/{name}/join1_infrastructure_condition_{name}.xlsx'
outputFile_join2 = f'../../../../Data/modified/joins/{name}/join2_infrastructure_condition_studentNumber_{name}.xlsx'
outputFile_join3 = f'../../../../Data/modified/joins/{name}/join3_infrastructure_condition_studentNumber_counts_{name}.xlsx'
outputFile_join4 = f'../../../../Data/modified/joins/{name}/join4_infrastructure_condition_studentNumber_counts_revenue_{name}.xlsx'

print('Joining infrastructure with condition')
join1 = join1_infrastructure_condition(df1, df2, withoutDifferentBuildings, save=True, outputPath=outputFile_join1)
print('Joining studentNumber')
join2 = join2_studentNumber(df3, join1, withDifferentBuildings, save=True, outputPath=outputFile_join2)
print('Joining school_building')
join3 = join3_school_building(join2, withoutDifferentBuildings, save=True, outputPath=outputFile_join3)
print('Joining revenue')
join4 = join4_revenue(join3, df4, withDifferentBuildings, save=True, outputPath=outputFile_join4)
