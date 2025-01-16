import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chapter2.DataPreprocessing.concatenateTables.utils.preprocess_functions import preprocess_name


def join2_studentNumber(df1, df2, withDifferentBuildings, save=False, outputPath=None):
	df1 = df1.copy()
	df2 = df2.copy()

	if withDifferentBuildings:
		df1, df2 = df2, df1
		deleteSchools = ["სსიპ - ხობის მუნიციპალიტეტის სოფელ საჯიჯაოს №2 საჯარო სკოლა",
						 "სსიპ - წყალტუბოს მუნიციპალიტეტის სოფელ მექვენის საჯარო სკოლა",
						 "სსიპ - ტყიბულის მუნიციპალიტეტის სოფელ ოჯოლის საჯარო სკოლა",
						 "სსიპ - მარნეულის მუნიციპალიტეტის სოფელ თაზაქენდის №1 საჯარო სკოლა",
						 "სსიპ - თიანეთის მუნიციპალიტეტის სოფელ ჩეკურაანთგორის საჯარო სკოლა",
						 "სსიპ - დუშეთის მუნიციპალიტეტის სოფელ კაიშაურების საჯარო სკოლა",
						 "სსიპ - დუშეთის მუნიციპალიტეტის სოფელ ყვავილის საჯარო სკოლა",
						 "სსიპ - დუშეთის მუნიციპალიტეტის სოფელ ხანდოს საჯარო სკოლა",
						 "სსიპ - გორის მუნიციპალიტეტის სოფელ ქვეშის საჯარო სკოლა",
						 "სსიპ - გორის მუნიციპალიტეტის სოფელ ქვეშის საჯარო სკოლა (ქვემო არცევის კორპუსი)",
						 "სსიპ - ამბროლაურის მუნიციპალიტეტის სოფელ ნამანევის საჯარო სკოლა",
						 "სსიპ - ამბროლაურის მუნიციპალიტეტის სოფელ ხოტევის საჯარო სკოლა",
						 "სსიპ - ახალქალაქის მუნიციპალიტეტის სოფელ აფნიის საჯარო სკოლა",
						 "სსიპ - აბაშის მუნიციპალიტეტის სოფელ კვათანის საჯარო სკოლა"]

		renameSchools = {
			"სსიპ - ტყიბულის მუნიციპალიტეტის სოფელ მუხურას №2 საჯარო სკოლა": "სსიპ - ტყიბულის მუნიციპალიტეტის სოფელ მუხურის №2 საჯარო სკოლა"}

		df1 = df1[~df1['სკოლის სახელწოდება'].isin(deleteSchools)]

		# Rename the schools based on the renameSchools dictionary
		df1['სკოლის სახელწოდება'] = df1['სკოლის სახელწოდება'].replace(renameSchools)

	df1['სკოლის სახელწოდება_cleaned'] = df1['სკოლის სახელწოდება'].apply(lambda x: preprocess_name(x))
	df2['სკოლის სახელწოდება_cleaned'] = df2['სკოლის სახელწოდება'].apply(lambda x: preprocess_name(x))

	vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 24))
	all_names = pd.concat([df1['სკოლის სახელწოდება_cleaned'], df2['სკოლის სახელწოდება_cleaned']])
	tfidf_matrix = vectorizer.fit_transform(all_names)

	# Compute cosine similarity
	similarity_matrix = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])

	# Find the best match for each school name
	threshold = 0.1  # Higher threshold for stricter matching

	if withDifferentBuildings:

		columns_to_update = [
			('სკოლის სახელწოდება რიცხვში', 'სკოლის სახელწოდება'),
			('კოდი (ცხრანიშნა) რიცხვში', 'კოდი (ცხრანიშნა)'),
			('კოდი (ოთხნიშნა) რიცხვში', 'კოდი (ოთხნიშნა)'),
			('მოსწავლეთა რაოდენობა', 'მოსწავლეთა რაოდენობა')
		]

		for df1_col, df2_col in columns_to_update:
			df1[df1_col] = [
				df2[df2_col].iloc[i] if sim.max() > threshold else None
				for sim, i in zip(similarity_matrix, similarity_matrix.argmax(axis=1))
			]

		df1['მსგავსების ქულა'] = [
			similarity_matrix[idx, similarity_matrix[idx].argmax()]
			for idx in range(len(similarity_matrix))
		]

		"""
		# rows = df1[df1['მსგავსების ქულა'] < 0.98]
		# differentCodes = rows[rows['კოდი (ცხრანიშნა)'] != rows['კოდი (ცხრანიშნა) რიცხვში']]
		# We can see that there are some schools that have a similarity of 1.0 but the codes are different
		# We manually checked the schools and found that the schools are in fact same with erros in codes
		# hence we can update the codes of the schools with the codes of the schools in the second dataset
		# We don't need to remove any rows because the tfidf vectorizer is already doing a good job
		"""

		new_order = ['რეგიონი', 'ქალაქი/მუნიციპალიტეტი',
					 'სკოლის სახელწოდება',
					 'სკოლის სახელწოდება რიცხვში',
					 'კოდი (ცხრანიშნა) რიცხვში',
					 'კოდი (ოთხნიშნა) რიცხვში',
					 'მოსწავლეთა რაოდენობა',
					 'შენობის მდგომარეობა',
					 'საჭიროა თუ არა ახალი სკოლის აშენება',
					 'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)', 'ეზოს ფართობი (კვ.მ)',
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
					 'ინდივიდუალური  გათბობა (გაზზე, დიზელზე, ქვანახშირზე, შეშაზე, ელექტროენერგიაზე, ბრიკეტები, სხვა)',
					 ]
		df1 = df1[new_order]
		df1.rename(
			columns={'კოდი (ცხრანიშნა) რიცხვში': 'კოდი (ცხრანიშნა)', 'კოდი (ოთხნიშნა) რიცხვში': 'კოდი (ოთხნიშნა)'},
			inplace=True)

		df1['სკოლის სახელწოდება'] = df1['სკოლის სახელწოდება'].apply(
			lambda x: preprocess_name(x, remove_punctuation=False, remove_english=False))

		df1['სკოლის სახელწოდება რიცხვში'] = df1['სკოლის სახელწოდება რიცხვში'].apply(
			lambda x: preprocess_name(x, remove_punctuation=False, remove_english=False))


	else:
		df1['სკოლის სახელწოდება ინფრასტრუქტურაში'] = [
			df2['სკოლის სახელწოდება'].iloc[i] if sim.max() > threshold else None
			for sim, i in zip(similarity_matrix, similarity_matrix.argmax(axis=1))
		]

		# Fetch 'კოდი (ცხრანიშნა)' from df2 based on matches
		df1['კოდი (ცხრანიშნა) ინფრასტრუქტურაში'] = [
			df2['კოდი (ცხრანიშნა)'].iloc[i] if sim.max() > threshold else None
			for sim, i in zip(similarity_matrix, similarity_matrix.argmax(axis=1))
		]
		cols = ['შენობის მდგომარეობა',
				'საჭიროა თუ არა ახალი სკოლის აშენება',
				'შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)', 'ეზოს ფართობი (კვ.მ)',
				'სართულების რაოდენობა', 'ფასადის მდგომარეობა', 'სახურავის მდგომარეობა',
				'პანდუსის მდგომარეობა', 'გარე კარ-ფანჯრის მდგომარეობა',
				'ცენტრალური გათბობის მდგომარეობა',
				'ცენტრალური გათბობის მოწყობის ან სრული რეაბილიტაციის წელი',
				'ელექტროობის მდგომარეობა',
				'წყალგაყვანილობის მდგომარეობა',
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
				'სააქტო დარბაზის მდგომარეობა',
				'ეზოს კეთილმოწყობა', 'ფანჯრების ტიპი', 'რით თბება სკოლის შენობა',
				'ცენტრალური გათბობა - (გაზი, დიზელი, ქვანახშირი, შეშა, ელექტროენერგია, ბრიკეტები, მზის სისტემა, სხვა)',
				'ინდივიდუალური  გათბობა (გაზზე, დიზელზე, ქვანახშირზე, შეშაზე, ელექტროენერგიაზე, ბრიკეტები, სხვა)']

		for col in cols:
			df1[col] = [
				df2[col].iloc[i] if sim.max() > threshold else None
				for sim, i in zip(similarity_matrix, similarity_matrix.argmax(axis=1))
			]

		df1['მსგავსების ქულა'] = [
			similarity_matrix[idx, similarity_matrix[idx].argmax()]
			for idx in range(len(similarity_matrix))
		]

		# Verify the resulting DataFram
		rows = df1[df1['მსგავსების ქულა'] < 0.98]
		deleteCodes = rows[rows['კოდი (ცხრანიშნა)'] != rows['კოდი (ცხრანიშნა) ინფრასტრუქტურაში']]

		df1 = df1.drop(deleteCodes.index)
		df1 = df1.drop(columns=['სკოლის სახელწოდება ინფრასტრუქტურაში',
								'მსგავსების ქულა',
								'კოდი (ცხრანიშნა) ინფრასტრუქტურაში',
								'სკოლის სახელწოდება_cleaned'])

		df1['სკოლის სახელწოდება'] = df1['სკოლის სახელწოდება'].apply(
			lambda x: preprocess_name(x, remove_punctuation=False, remove_english=False))
	if save:
		df1.to_excel(outputPath, index=False)
	return df1

