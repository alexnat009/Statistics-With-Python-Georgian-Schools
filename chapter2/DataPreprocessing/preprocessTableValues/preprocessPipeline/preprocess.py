import pandas as pd

from chapter2.DataPreprocessing.preprocessTableValues.categoricalPreprocessing import categoricalPreprocess_main
import os
from chapter2.DataPreprocessing.preprocessTableValues.nullValuePreparation import nullValuePreprocess_main
from chapter2.DataPreprocessing.preprocessTableValues.outlierDetection.outlierDetection import outlierDetection_main
from chapter2.DataPreprocessing.preprocessTableValues.revenuePreprocessings import revenuePreprocessings_main

withDifferentBuildings = False
name = "withDifferentBuildings" if withDifferentBuildings else "withoutDifferentBuildings"

# Construct absolute paths
base_dir = os.path.abspath("../../../../Data/modified")  # Adjust this based on your directory structure
inputFilePath = os.path.join(base_dir,
							 f"joins\\{name}\\join4_infrastructure_condition_studentNumber_counts_revenue_{name}.xlsx")
outputFilePath1 = os.path.join(base_dir,
							   f"preprocesses\\categoricalFormat\\preprocess1_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl")
outputFilePath2 = os.path.join(base_dir,
							   f"preprocesses\\DropNA\\preprocess2_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl")
outputFilePath3 = os.path.join(base_dir,
							   f"preprocesses\\revenueFormat\\preprocess3_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl")
outputFilePath4_1 = os.path.join(base_dir,
								 f"preprocesses\\dropOutliers\\preprocess4_infrastructure_condition_studentNumber_counts_revenue_join_{name}.pkl")
outputFilePath4_2 = os.path.join(base_dir, f"finalDatabase\\{name}\\finalDatabase_{name}")
preprocess0 = pd.read_excel(inputFilePath)
preprocess1 = categoricalPreprocess_main(preprocess0, outputFilePath1, save=False)
preprocess2 = nullValuePreprocess_main(preprocess1, outputFilePath2, save=False)
preprocess3 = revenuePreprocessings_main(preprocess2, outputFilePath3, save=False)
preprocess4 = outlierDetection_main(preprocess3, outputFilePath4_1, name, save=False)
# preprocess4.to_pickle(outputFilePath4_2 + ".pkl")
# preprocess4.to_excel(outputFilePath4_2 + ".xlsx", index=False)
