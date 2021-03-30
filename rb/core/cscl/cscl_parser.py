import csv
import json
from datetime import datetime
from typing import Dict, List

import xmltodict
from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.community import Community
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.participant import Participant
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.rblogger import Logger
from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration


logger = Logger.get_logger()

CONTRIBUTIONS_KEY = 'contributions'
ID_KEY = 'id'
PARENT_ID_KEY = 'parent_id'
TIMESTAMP_KEY = 'timestamp'
TEXT_KEY = 'text'
USER_KEY = 'user'
CONV_ID = 'conv_id'

JSONS_PATH = './jsons/'

FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f %Z", 
    "%Y-%m-%d %H:%M:%S %Z", 
    "%Y-%m-%d %H:%M %Z",
    "%Y-%m-%d %H:%M:%S.%f", 
    "%Y-%m-%d %H:%M:%S", 
    "%Y-%m-%d %H:%M",
    "%H.%M.%S",
    "%H:%M:%S",
]

def get_json_from_json_file(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename, "rt", encoding='utf-8') as json_file:
        contribution_list_json = json.load(json_file)
        for cnt in contribution_list_json:
            contribution = {
                ID_KEY: cnt['genid'],
                PARENT_ID_KEY: cnt["ref"],
                TIMESTAMP_KEY: int(float(cnt['time'])),
                USER_KEY: cnt["nickname"],
                TEXT_KEY: cnt["text"],
            }
            contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def get_json_from_csv(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first = True			

        for row in csv_reader:
            if first:
                first = False
            else:
                contribution = dict()

                contribution[ID_KEY] = row[0]
                contribution[PARENT_ID_KEY] = row[1]
                contribution[USER_KEY] = row[2]
                contribution[TEXT_KEY] = row[3]
                contribution[TIMESTAMP_KEY] = row[4]

                contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def parse_large_csv(filename: str) -> Dict:
    conversation_thread = dict()
    contribution_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first = True			

        for row in csv_reader:
            if first:
                first = False
            else:
                contribution = dict()

                contribution[ID_KEY] = row[0]
                contribution[PARENT_ID_KEY] = row[1]
                if row[1] == '':
                    contribution[PARENT_ID_KEY] = '-1'

                contribution[USER_KEY] = row[2]
                contribution[TEXT_KEY] = row[6]

                contribution[TIMESTAMP_KEY] = read_date(row[3])

                contribution_list.append(contribution)

    conversation_thread[CONTRIBUTIONS_KEY] = contribution_list

    return conversation_thread

def read_date(date) -> datetime:
    for date_format in FORMATS:
        try:
            return datetime.strptime(date, date_format)
        except:
            pass
    

def load_from_xml(filename: str, diacritics_model: DiacriticsRestoration = None) -> Dict:
	with open(filename, "rt", encoding="utf-8") as f:
		my_dict=xmltodict.parse(f.read())
		if "corpus" in my_dict:
			my_dict = my_dict["corpus"]
		turns = my_dict["Dialog"]["Body"]["Turn"]
		if not isinstance(turns, List):
			turns = [turns]
		contributions = [
			{
				ID_KEY: int(utterance["@genid"]) - 1,
				PARENT_ID_KEY: int(utterance["@ref"]) - 1 if utterance["@ref"] else -1,
				TIMESTAMP_KEY: read_date(utterance["@time"]),
				USER_KEY: turn["@nickname"],
				TEXT_KEY: diacritics_model.process_string(utterance["#text"], mode="replace_missing") if diacritics_model else utterance["#text"],
			}
			for turn in turns
			for utterance in (turn["Utterance"] if isinstance(turn["Utterance"], List) else [turn["Utterance"]])
            if "#text" in utterance
		]

		return {CONTRIBUTIONS_KEY: contributions, CONV_ID: my_dict["Dialog"]["@team"]}


def export_individual_statistics(participants: List[Participant], filename: str):
	first = True

	cscl_keys = [CNAIndices.CONTRIBUTIONS_SCORE, CNAIndices.SOCIAL_KB, CNAIndices.NO_CONTRIBUTION, CNAIndices.OUTDEGREE, CNAIndices.INDEGREE,
				CNAIndices.NO_NEW_THREADS, CNAIndices.NEW_THREADS_OVERALL_SCORE, CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB,
				CNAIndices.AVERAGE_LENGTH_NEW_THREADS]

	evaluate_interaction(conv)
	evaluate_involvement(conv)
	evaluate_textual_complexity(conv)
	perform_sna(conv, False)
	with open(filename, 'w') as f:
		for p in participants:
			indices = p.indices
			keys = ['participant'] + [str(k.value) for k in CNAIndices]
			values = [p.get_id()] + [str(p.get_index(k)) for k in cscl_keys]

			if first:
				f.write(','.join(keys) + '\n')
				first = False

			f.write(','.join(values) + '\n')
		f.flush()
		f.close()

def export_textual_complexity(participants: List[Participant], filename: str):
	first = True

	with open(filename, 'w') as f:
		for p in participants:
			indices = p.textual_complexity_indices
			keys = ['participant'] + [str(k) for k in indices.keys()]
			values = [p.get_id()] + [str(indices[k]) for k in indices.keys()]

			if first:
				f.write(','.join(keys) + '\n')
				first = False

			f.write(','.join(values) + '\n')
		f.flush()
		f.close()
