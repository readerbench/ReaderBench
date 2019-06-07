def get_pronoun(tag):
	#Pp1-sn--------s
	return tag[2], tag[4], tag[3]

def get_verb(tag):
	if len(tag) < 6:
		number = ""
	else:
		number = tag[5]
	return tag[4], number

def get_noun(tag):
	return '3', tag[3], tag[2]

def get_adjective(tag):
	return "-", tag[4], tag[3]

def get_numeral(tag):
	return "-", tag[3], tag[2]

def get_person_number_gender(spn):
	person = ""
	number = ""
	gender = ""
	if spn[0] == "3":
		person = "a 3-a"
	else:
		person = spn[0]
	if spn[1] == "s":
		number = "singular"
	else:
		number = "plural"
	if spn[2] == "f":
		gender = "feminin"
	else:
		gender = "masculin"
	return person, number, gender