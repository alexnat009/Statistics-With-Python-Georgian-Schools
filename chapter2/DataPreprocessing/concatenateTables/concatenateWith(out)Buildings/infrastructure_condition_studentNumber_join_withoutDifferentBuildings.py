import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess_functions import preprocess_name

inputFile1 = '../../../Data/modified/modified მოსწავლეთა რაოდენობა.xlsx'
inputFile2 = '../../../Data/modified/joins/infrastructure_condition_join_withoutDifferentBuildings.xlsx'
outputFile = '../../../Data/modified/joins/infrastructure_condition_studentNumber_join_withoutDifferentBuildings.xlsx'

df1 = pd.read_excel(inputFile1)
df2 = pd.read_excel(inputFile2)

# Apply preprocessing
df1['სკოლის სახელწოდება_cleaned'] = df1['სკოლის სახელწოდება'].apply(lambda x: preprocess_name(x))
df2['სკოლის სახელწოდება_cleaned'] = df2['სკოლის სახელწოდება'].apply(lambda x: preprocess_name(x))

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 16))  # Character n-grams
all_names = pd.concat([df1['სკოლის სახელწოდება_cleaned'], df2['სკოლის სახელწოდება_cleaned']])
tfidf_matrix = vectorizer.fit_transform(all_names)

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix[:len(df1)], tfidf_matrix[len(df1):])

# Find the best match for each school name
threshold = 0.1  # Higher threshold for stricter matching

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

# df1.to_excel(outputFile, index=False)
