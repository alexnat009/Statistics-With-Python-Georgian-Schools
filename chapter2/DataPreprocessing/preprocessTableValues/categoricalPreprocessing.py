import numpy as np
import pandas as pd
import re
from pandas.api.types import CategoricalDtype


# noinspection SpellCheckingInspection
def remove_bad_rows(df):
	bad_rows = [
		"სსიპ - ქალაქ თბილისის №84 საჯარო სკოლა",
		"სსიპ - ქალაქ ბათუმის №16 საჯარო სკოლა",
		"სსიპ - ხულოს მუნიციპალიტეტის რუსლან მელაძის სახელობის სოფელ ქვემო ვაშლოვანის საჯარო სკოლა",
		"სსიპ - გურჯაანის მუნიციპალიტეტის სოფელ ბაკურციხის საჯარო სკოლა",
		"სსიპ - რუსლან მელაძის სახელობის ხულოს მუნიციპალიტეტის სოფელ ქვემო ვაშლოვანის საჯარო სკოლა"
	]

	df = df[~df['სკოლის სახელწოდება'].isin(bad_rows)]
	return df


def yard_area(value):
	numbers = re.findall(r'\d+', str(value))
	if pd.isnull(value):
		return value
	elif value == '13 360':
		return int(numbers[0] + numbers[1])
	elif numbers:
		return int(numbers[0])


def building_area(value):
	numbers = re.findall(r'\d+', str(value))
	if pd.isnull(value):
		return value

	if len(numbers) > 2:
		return pd.Series(numbers).astype(np.int_).nlargest(np.sqrt(len(numbers)).astype(int)).sum()
	else:
		return int(numbers[0])


def floor_number(value):
	# Mapping of number words to digits
	word_to_digit = {
		'ერთ': '1',
		'ორ': '2',
		'სამ': '3',
		'III': '3',
		'IV': '4'
	}
	if pd.isnull(value):  # Handle NaN values
		return value

	# Convert the value to a string for processing
	value = str(value)

	# Replace number words with corresponding digits
	for word, digit in word_to_digit.items():
		value = re.sub(rf'\b{word}\b', digit, value)  # Replace whole words only

	# Extract all numeric characters
	digits = re.findall(r'\d', value)
	if digits:
		# Find the highest digit and return it as an integer
		return int(max(digits))
	return None  # If no digits are found, return None


def heating_system_date(value):
	if pd.isnull(value):  # Handle NaN values
		return value
	# Extract numbers from the string using regex
	numbers = re.findall(r'\d+', str(value))
	if numbers:
		# Combine all numbers found and get the last four digits
		last_four = ''.join(numbers)[-4:]
		return int(last_four)
	return None  # Return None if no numbers are found


# noinspection SpellCheckingInspection
def heating_system_type(value):
	value = str(value)
	if pd.isnull(value) or value == 'არ არსებობს' or value == 'სხვა':  # Handle NaN values
		return value
	# remove punctuation
	value = re.sub(r'[^\w\s]', ' ', value)
	result = ''
	if bool(re.search(r'გა.ი|გ.ზ|ცენტრ|ბუნებრივი აირი', value)):
		result = result + 'გაზი '
	if bool(re.search(r'[შს]ე[შს]', value)):
		result = result + 'შეშა '
	if bool(re.search(r'ბრ[იეუკ][კლი]', value)):
		result = result + 'ბრიკეტი '
	if bool(re.search(r'დიზ', value)):
		result = result + 'დიზელი '
	if bool(re.search(r'ენერ|დენ|ექტრო|ელ გა', value)):
		result = result + 'ელექტროენერგია '
	if bool(re.search(r'ხშირ', value)):
		result = result + 'ქვანახშირი '
	if bool(re.search(r'ნაჭ', value)):
		result = result + 'თხილის ნაჭუჭი '
	if len(result) != 0:
		# remove the extra white space
		return result[:-1]
	return None


# noinspection SpellCheckingInspection
filters = {
	"შენობის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'მიმდინარეობს რეაბილიტაცია': ['მიმდინარეობს სრული რეაბ']
		},
		"datatype": CategoricalDtype(categories=[
			'სკოლას არ აქვს საკუთარი შენობა',
			'მიმდინარეობს მშენებლობა',
			'არ არის რეაბილიტირებული',
			'ნაწილობრივ რეაბილიტირებულია',
			'მიმდინარეობს რეაბილიტაცია',
			'სრულად რეაბილიტირებულია',
			'ახალაშენებულია'
		], ordered=True)
	},
	"საჭიროა თუ არა ახალი სკოლის აშენება": {
		"function": lambda x: x,
		'map_values': {
			'არა': ['არ არის აუცილებელი', 'ნაწილობრივ რეაბილიტირებულია', 'არ არის საჭირო ახალი შენობა'],
			'საჭიროა სხვა მიზეზით': ['კი'],
			'საჭიროა სკოლას არ აქვს შენობა': ['საჭიროა-სკოლას არ აქვს შენობა']
		},
		"datatype": CategoricalDtype(categories=[
			'არა', 'საჭიროა ფართის უკმარისობის გამო',
			'საჭიროა ავარიულობის გამო',
			'საჭიროა სხვა მიზეზით',
			'საჭიროა სკოლას არ აქვს შენობა'
		])
	},
	"შენობის ფართი, რომელიც გამოყენებაშია (კვ.მ)": {
		'function': building_area,
		'map_values': {},
		"datatype": pd.Int64Dtype()
	},
	"ეზოს ფართობი (კვ.მ)": {
		'function': yard_area,
		'map_values': {},
		"datatype": pd.Int64Dtype()
	},
	"სართულების რაოდენობა": {
		'function': floor_number,
		'map_values': {},
		"datatype": pd.Int64Dtype()

	},
	"ფასადის მდგომარეობა": {
		'function': lambda x: x,
		'map_values': {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ნაწილობრივ რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებულია'],
			'ცუდი': [
				'მთლიანად სარეაბილიტაციო', 'მთლიანად სარებილიტციო',
				'სარეამილიტაციო', 'სარეაბილიტაციო', 'სრულად სარეაბილიტაციო',
				'მთლიანად სარეაბილიტაციოა'
			]
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"სახურავის მდგომარეობა": {
		'function': lambda x: x,
		'map_values': {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ნაწილობრივ რიაბილიტირებულია', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['მთლიანად სარეაბილიტაციო', 'გამოსაცვლელია', 'სარეაბილიტაციო']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"პანდუსის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ნორმალური'],
			'ცუდი': ['ნორმებში არ ზის'],
			'არ არსებობს': ['არ გვაქვს', 'არ არსებობს', 'არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"გარე კარ-ფანჯრის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული', 'ახალია '],
			'დამაკმაყოფილებელი': ['ნაწილობრივ რეაბილიტირებული', 'ნაწილობრივ სარეაბილიტაციო'],
			'ცუდი': ['მთლიანად სარეაბილიტაციო'],
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ცენტრალური გათბობის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ნორმალური', 'ნაწილობრივ თბება', 'საშუალო'],
			'ცუდი': ['მთლიანად სარეაბილიტაციო', 'სარეაბილიტაციო'],
			'არ არსებობს': ['არ არის', 'არ არსეობს'],
			np.nan: ['ბრიკეტები', 'გაზი', 'სხვა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ცენტრალური გათბობის მოწყობის ან სრული რეაბილიტაციის წელი": {
		"function": heating_system_date,
		"map_values": {},
		"datatype": pd.Int64Dtype()
	},
	"ელექტროობის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': [
				'ნორმალური', 'ძირითადად რეაბილიტირებულია',
				'ძირითადად რეაბილიტირებულია, სარეაბილიტაციოა ძირითად და დამხმარე შენობას შორის '
				'გადასასვლელი ხიდი.', 'დამაკმაყოფილებებლი', 'ნაწილობრივ რეაბილიტირებული'
			],
			'ცუდი': ['ნორმებში არ ზის', 'მთლიანად სარეაბილიტაციოა', 'მთლიანად სარეაბილიტაციო', 'სარეაბილიტაციო']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"წყალგაყვანილობის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული', 'რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ნაწილობრივ რეაბილიტირებული', 'ნორმალური', 'ნაწილობრივ სარეაბილიატაციოა'],
			'ცუდი': ['მთლიანად სარეაბილიტაციო']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"სველი წერტილების (საპირფარეშოების) ზოგადი მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': [
				'ნაწილობრივ რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული ', 'ნორმალური',
				'ნაწილობრივ სარეაბილიტაციოა'
			],
			'ცუდი': ['მთლიანად სარეაბილიტაციო']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"საპირფარეშო ოთახების რაოდენობა": {
		"function": lambda x: x,
		"map_values": {},
		"datatype": pd.Int64Dtype()
	},
	"საკლასო ოთახების რაოდენობა": {
		"function": lambda x: x,
		"map_values": {},
		"datatype": pd.Int64Dtype()
	},
	"I სართულის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"II სართულის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"III სართულის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"IV სართულის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული'],
			'არ არსებობს': ['არა', 'არა ქვს']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ადმინისტრაციის ოთახების მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"კომპიუტერების ოთახის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ბიბლიოთეკის ოთახის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ბუფეტის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა', 'არ არსებობს ']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ფიზიკის/ქიმიის/ბიოლოგიის კაბინეტ-ლაბორატორიის (სამივე ლაბორატორია როცა ერთ ოთახშია) მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არა', 'არ არსებობს ', 'არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ფიზიკის კაბინეტ-ლაბორატორიის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ქიმიის კაბინეტ-ლაბორატორიის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ბიოლოგიის კაბინეტ-ლაბორატორიის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული'],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო'],
			'არ არსებობს': ['არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"გარე სპორტული მოედნების მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული', 'I სპორტული კარგი-2023, II სპორტული მოედანი-სარეაბილიტაციო'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული', 'დამაკმაყოფილებელი '],
			'ცუდი': ['არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო', 'არადამაკმაყოფილებელი'],
			'არ არსებობს': ['არ არის']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"სპორტული დარბაზების მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {
			'დამაკმაყოფილებელი': ['ნაწილობრივ სარეაბილიტაციო'],
			'ცუდი': ['სარეაბილიტაციო'],
			'არ არსებობს': ['არ აქვს', 'არა']},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"სპორტული დარბაზების რაოდენობა": {
		"function": lambda x: x,
		"map_values": {
			0: ['არ არსებობს', 'არა'],
			1: ['ერთი', 'ცუდი']
		},
		"datatype": pd.Int64Dtype()
	},
	"სააქტო დარბაზის მდგომარეობა": {
		"function": lambda x: x,
		"map_values": {'არ არსებობს': ['არა']},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ეზოს კეთილმოწყობა": {
		"function": lambda x: x,
		"map_values": {
			'კარგი': ['სრულად რეაბილიტირებული'],
			'დამაკმაყოფილებელი': ['ძველი რეაბილიტირებული', 'ნაწილობრივ რეაბილიტირებული', 'რეაბილიტირებული'],
			'ცუდი': [
				'არ არის რეაბილიტირებული', 'სარეაბილიტაციო', 'მთლიანად სარეაბილიტაციო',
				'არ არის რეაბილიტირებულ'
			],
			'არ არსებობს': ['არა']
		},
		"datatype": CategoricalDtype(
			categories=['არ არსებობს', 'ცუდი', 'დამაკმაყოფილებელი', 'კარგი'], ordered=True)
	},
	"ფანჯრების ტიპი": {
		"function": lambda x: x,
		"map_values": {
			'მეტალოპლასტმასი': ['მეტალოპლასტმასი ', 'მეტალოპლასმასი', 'მეტალოპლასტმ'],
			'მეტალოპლასტმასი და ხე': ['მეტალოპლასმასი-ხე'],
			'მეტალოპლასტმასი და ალუმინი': ['მეტალოპლასმასი და ალუმინი']
		},
		"datatype": CategoricalDtype(
			categories=[
				'მეტალოპლასტმასი', 'მეტალოპლასტმასი და ალუმინი',
				'ხე',
				'მეტალოპლასტმასი და ხე', 'იზოალუმინი',
				'სხვა', 'იზოალუმინი და მეტალოპლასტმასი',
				'ალუმინი და ხე'
			])
	},
	"რით თბება სკოლის შენობა": {
		"function": lambda x: x,
		"map_values": {
			'ცენტრალური': ['ცენტრალური გათბობა', 'გაზი', 'დიზელი', 'თხილის ნაჭუჭი'],
			'ელექტროღუმელები': [
				'შენობა არის ძალიან მაღალჭერიანი სქელი კედლებით და ამიტომვერ თბება, ვამატებთ ელექტრო '
				'ღუმელებს.          2012წ'
			],
			'შეშის ღუმელები': ['ბრიკეტები']
		},
		"datatype": CategoricalDtype(
			categories=[
				'ცენტრალური', 'შეშის ღუმელები',
				'ელექტროღუმელები',
				'ინდივიდუალური გაზის გამათბობლები', 'სხვა'
			])
	},
	"ცენტრალური გათბობა - (გაზი, დიზელი, ქვანახშირი, შეშა, ელექტროენერგია, ბრიკეტები, მზის სისტემა, სხვა)": {
		"function": heating_system_type,
		"map_values": {
			'არ არსებობს': [
				'არ არის', 'არა', 'არ გვაქვს', 'არ გვაქვს ცენტრალური გათბობა',
				'ცენტრალური გათბობა არ არის',
				'ინდივიდუალური გათბობა არ არის',
				' არ არსებობს', 'ცენტრალური გათბობა არაა', '_', 0
			]
		},
		"datatype": CategoricalDtype(
			categories=[
				'გაზი',
				'შეშა', 'შეშა ბრიკეტი',
				'არ არსებობს', 'ელექტროენერგია',
				'შეშა ელექტროენერგია', 'ქვანახშირი',
				'დიზელი', 'ბრიკეტი',
				'ბრიკეტი ქვანახშირი', 'შეშა ბრიკეტი ქვანახშირი',
				'გაზი შეშა', 'შეშა დიზელი',
				'შეშა ქვანახშირი', 'შეშა დიზელი ქვანახშირი',
				'გაზი ბრიკეტი', 'გაზი ელექტროენერგია',
				'ბრიკეტი ელექტროენერგია', 'შეშა ბრიკეტი ელექტროენერგია',
				'გაზი დიზელი', 'გაზი შეშა დიზელი',
				'გაზი თხილის ნაჭუჭი', 'შეშა დიზელი ელექტროენერგია',
				'სხვა'
			])
	},
	"ინდივიდუალური  გათბობა (გაზზე, დიზელზე, ქვანახშირზე, შეშაზე, ელექტროენერგიაზე, ბრიკეტები, სხვა)": {
		"function": heating_system_type,
		"map_values": {
			'არ არსებობს': [
				'არ არის', 'არა', 'არ გვაქვს', 'არ გვაქვს ცენტრალური გათბობა',
				'ცენტრალური გათბობა არ არის',
				'ინდივიდუალური გათბობა არ არის',
				' არ არსებობს', 'ცენტრალური გათბობა არაა', '_', 0
			]
		},
		"datatype": pd.StringDtype()
	},
	"შენობების რაოდენობა": {
		"function": lambda x: x,
		"map_values": {},
		"datatype": pd.Int64Dtype()
	},
	"ცვლების რაოდენობა": {
		"function": lambda x: x,
		"map_values": {},
		"datatype": pd.Int64Dtype()
	},
}


def map_column_values(df, column, mapping_dict, func, datatype):
	if mapping_dict is None:
		mapping_dict = {}
	reverse_mapping = {v: k for k, values in mapping_dict.items() for v in values}

	# Apply the mapping to the column
	df[column] = df[column].apply(lambda x: reverse_mapping.get(x, x))
	df[column] = df[column].apply(func)
	if datatype == "Int64":
		df[column] = pd.to_numeric(df[column]).astype(pd.Int64Dtype())
	else:
		df[column] = df[column].astype(datatype)
	return df


def categoricalPreprocess_main(df, output_path, save=False):
	df = df.copy()
	df = remove_bad_rows(df)
	for i, (key, value) in enumerate(filters.items()):
		df = map_column_values(df, key, value["map_values"], value["function"], value["datatype"])
	if save:
		df.to_pickle(output_path)
		df.to_excel(output_path.replace('.pkl', '.xlsx'), index=False)
	return df
