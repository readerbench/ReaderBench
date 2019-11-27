from rb.processings.ro_corrections.get_exceptions import *

def get_pronoun(tag):
	return tag[2], tag[4], tag[3]

def get_verb(tag):
	if len(tag) < 6:
		number = ""
	else:
		number = tag[5]
	return tag[4], number

def get_noun(tag):
	#not enough tag details
	if tag != "Np":
		return '3', tag[3], tag[2]

def get_adjective(tag):
	return "-", tag[4], tag[3]

def get_numeral(tag):
	return "-", tag[3], tag[2]

def get_unstated_article(tag):
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


#word could be a noun or a predicate
def get_multiple_subject(word, relation, exceptions):
	subjects = []
	if relation == "conj":
		subjects = [word]
	for child in word.children:
		if child.dep_ == relation:
			subjects.append(child)

	persons = are_describing_persons(subjects, exceptions)
	if len(persons) != 0:
		p = get_person_for_multiple_subject(persons)
		if all_feminine_gender(persons):
			return p, "p", "f"
		else :
			return p, "p", "m"

	all_nouns = True
	for subject in subjects:
		if subject.tag_[:2] != "Nc" and subject.tag_[:2] != "Np":
			all_nouns = False
			break
	if all_nouns:
		#if there is at least one subject at gender masculin and plural -> near noun is the influent, else will be feminin plural
		if at_least_one_plural_masculin_noun(subjects):
			near_subject = subjects[len(subjects) - 1]
			(person, number, gender) = get_noun(near_subject.tag_)
			return "3", "p", gender
		if at_least_one_feminin_noun(subjects):
			return "3", "p", "f"
		return "3", "p", "m"


def all_feminine_gender(persons):
	for person in persons:
		if person.tag_[:2] == "Nc" or person.tag_[:2] == "Np":
			nounOrPronoun = get_noun(person.tag_)
		elif person.tag_[0] == "P":
			nounOrPronoun = get_pronoun(person.tag_)
		if nounOrPronoun[2] == "m" or nounOrPronoun[2] == "-":
			return False
	return True

def get_person_for_multiple_subject(persons):
	first = False
	second = False
	for person in persons:
		if person.tag_[:2] == "Nc" or person.tag_[:2] == "Np":
			nounOrPronoun = get_noun(person.tag_)
		elif person.tag_[0] == "P":
			nounOrPronoun = get_pronoun(person.tag_)
		if nounOrPronoun[0] == "1":
			first = True
			break
		if nounOrPronoun[0] == "2":
			second = True
	if first:
		return "1"
	elif second: 
		return "2"
	else:
		return "3"


def at_least_one_plural_masculin_noun(subjects):
	for subject in subjects:
		(person, number, gender) = get_noun(subject.tag_)
		if number == "p" and gender == "m":
			return True

	return False

def at_least_one_feminin_noun(subjects):
	for subject in subjects:
		(person, number, gender) = get_noun(subject.tag_)
		if gender == "f":
			return True

	return False

#al that are in conjunction relation
def get_multiple_predicates(root):
	predicates = [root]
	for child in root.children:
		if child.dep_ == "conj" and has_own_subject(child) == False:
			predicates.append(child)

	return predicates

#because 2 sentences that are in coordinative relation will have verbs in conjunction, even if they have own subject
def has_own_subject(predicate):
	for child in predicate.children:
		if child.dep_ == "nsubj":
			return True
	return False
