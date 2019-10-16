#check if it is a multiple subject
def is_multiple_subject_linked_to_subject(subject):
	for child in subject.children:
		if child.dep_ == "conj":
			return True
	return False

def is_multiple_subject_linked_to_predicate(predicate):

	#this case is for personal pronoun, and not only, but much more nsubj
	subjects = []
	for child in predicate.children:
		if child.dep_ == "nsubj":
			subjects.append(child)

	if len(subjects) > 1:
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

#adjectives that are linked with the other through conjunctions
def is_in_adjective_noun_relation(adjective):
	if adjective.dep_ == "amod":
		return (True, adjective.head)
	if adjective.head.dep_ == "amod" and adjective.head.head.tag_[:2] == "Nc":
		return (True, adjective.head.head)
	return (False, None)
