import requests
import concurrent.futures


def join3_school_building(df1, withoutDifferentBuildings, save=False, outputPath=None):
	urlAllSchools = "https://skolebi.emis.ge/back/school/search?page=1&size=2086&schoolTypes=PUBLIC"
	urlSingleSchool = "https://skolebi.emis.ge/back/school/search/"

	school_column_name = 'სკოლის სახელწოდება' if withoutDifferentBuildings else 'სკოლის სახელწოდება რიცხვში'

	# Step 1: Get all schools from the first API call
	try:
		response = requests.get(urlAllSchools)
		response.raise_for_status()  # Check for HTTP request errors
		all_schools = response.json()['result']
		# Map school names to their IDs
		school_info_map = {school['schoolName']: school['id'] for school in all_schools}
	except requests.exceptions.RequestException as e:
		print(f"Error fetching schools data: {e}")
		school_info_map = {}

	# Step 2: Efficiently update the DataFrame with real building count by fetching details if missing
	df1['შენობების რაოდენობა'] = None
	df1['ცვლების რაოდენობა'] = None

	# Thread-safe result collection
	results = []

	# Function to fetch school details and return results
	def fetch_school_data(school_name_in_df1):
		school_id = school_info_map.get(school_name_in_df1, None)
		if not school_id:
			# print(f"School ID not found for {school_name_in_df1}")
			return school_name_in_df1, None, None  # No ID found

		retries = 3
		for attempt in range(retries):
			try:
				response_building = requests.get(f'{urlSingleSchool}{school_id}', timeout=30)
				response_building.raise_for_status()
				building_data = response_building.json()
				building_count = building_data.get('buildingCount', None)

				response_shift = requests.get(f'{urlSingleSchool}{school_id}/firstgradelimit', timeout=30)
				response_shift.raise_for_status()
				shift_data = response_shift.json()
				try:
					shift = max(item['shift'] for item in shift_data)
				except ValueError:
					print(f"Shift data not found for {school_name_in_df1}")
					shift = 1

				return school_name_in_df1, building_count, shift
			except requests.exceptions.RequestException as e:
				print(f"Error on attempt {attempt + 1} for {school_name_in_df1}: {e}")

		return school_name_in_df1, None, None  # Return None if still failing after retries

	# Use threads to fetch data

	with concurrent.futures.ThreadPoolExecutor() as executor:
		futures = [executor.submit(fetch_school_data, school_name) for school_name in df1[school_column_name]]
		for future in concurrent.futures.as_completed(futures):
			results.append(future.result())

	# Update DataFrame in a single pass after all threads complete
	for school_name, building_count, shift_count in results:
		df1.loc[df1[school_column_name] == school_name, ['შენობების რაოდენობა', 'ცვლების რაოდენობა']] = [
			building_count, shift_count]

	# Step 3: Manually handle errors and missing data
	school_dict = {'სსიპ - ბოლნისის მუნიციპალიტეტის დაბა კაზრეთის №2 საჯარო სკოლა': (1, 2),
				   'სსიპ - გარდაბნის მუნიციპალიტეტის სოფელ სართიჭალის №1 საჯარო სკოლა': (1, 1),
				   'სსიპ - ილია ჭავჭავაძის სახელობის ქალაქ ყვარლის №1 საჯარო სკოლა': (1, 1),
				   'სსიპ - კასპის მუნიციპალიტეტის სოფელ კოდისწყაროს საჯარო სკოლა': (1, 1),
				   'სსიპ - ქალაქ ლანჩხუთის №3 საჯარო სკოლა': (2, 1),
				   'სსიპ - ლანჩხუთის მუნიციპალიტეტის სოფელ აკეთის საჯარო სკოლა': (2, 2),
				   'სსიპ - სულხან-საბა ორბელიანის სახელობის ქალაქ ბოლნისის №1 საჯარო სკოლა': (1, 2),
				   'სსიპ - გალაკტიონ ტაბიძის სახელობის ქალაქ ვანის №2 საჯარო სკოლა': (1, 1),
				   'სსიპ - ქრისტეფორე III-ის სახელობის ხაშურის მუნიციპალიტეტის დაბა სურამის №4 საჯარო სკოლა': (3, 1),
				   'სსიპ - ხობის მუნიციპალიტეტის სოფელ საჯიჯაოს №1 საჯარო სკოლა': (3, 3),
				   'სსიპ - ხობის მუნიციპალიტეტის სოფელ ქვემო ქვალონის №2 საჯარო სკოლა': (1, 1),
				   'სსიპ - ცაგერის მუნიციპალიტეტის სოფელ აღვის საჯარო სკოლა': (1, 1)}

	# Step 4: Update corresponding columns in the DataFrame
	for school_name, (num_buildings, num_shifts) in school_dict.items():
		# Update the columns in the DataFrame for the corresponding school name
		df1.loc[df1[school_column_name] == school_name, ['შენობების რაოდენობა', 'ცვლების რაოდენობა']] = [
			num_buildings, num_shifts]

	# Save the updated DataFrame to a new Excel file
	if save:
		df1.to_excel(outputPath,
					 index=False)

	return df1