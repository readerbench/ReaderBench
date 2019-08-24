#check if predicative name is in the list with jobs, because it is right Fata este doctor.
def is_job_noun(spn, predicative_name_text, exceptions):
    if spn == ("3", "s", "f") or spn == ("-", "s", "f"):
        for job in exceptions["Jobs"]:
            if job.split(" ")[0].lower() == predicative_name_text.text.lower():
                return True
    return False


#if it is a collective subject, it is also correct Majoritatea au votat DA.
def is_collective_subject(subject, spv, exceptions):
	if subject.tag_[0] == "P":
		return False
	if subject.text.lower() in exceptions["Collective nouns"] and spv == ("3", "p"):
		return True
	return False

def are_describing_persons(subjects, exceptions):
	persons = []
	for subject in subjects:
		if subject.text.lower() in exceptions["Persons"]:
			persons.append(subject)
	return persons
