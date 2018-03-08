#dictionary for ages
ages = [{"Start year":411.0,"Era":"Devonian","Age":"Paleozoic"},
  {"Start year":443.7.0,"Era":"Ordovician","Age":"Paleozoic"}]

def read_csv_file
""read csv files of 'extinction' and 'impact sturctures'"""

Year = 420
Last_age = None
For age in ages:
	If last_age == None:
		Last_age = age

	If year < age["start year"]:
		#that means that this event is in the *last* age
		Correct_age = last_age
		Break

Era = correct_age["era"]
Period = correct_age["age"]
