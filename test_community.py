from rb.core.cscl.cscl_parser import load_from_xml
from rb.core.lang import Lang
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
from rb.similarity.vector_model_factory import get_default_model
from rb.processings.cscl.participant_evaluation import perform_sna, evaluate_interaction, evaluate_involvement, evaluate_textual_complexity
from rb.processings.cscl.community_time_processing import determine_subcommunities
from rb.cna.cna_graph import CnaGraph
import os
from datetime import datetime
from rb.core.cscl.cna_indices_enum import CNAIndices
import json
from rb.processings.cscl.community_processing import compute_new_threads_metrics
from rb.processings.keywords.keywords_extractor import extract_keywords
import csv
from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration

def transform_graph(graph):
    nodes_index = {
        node.get_id(): j
        for j, node in enumerate(graph.nodes())
    }
    return {
        "nodes": [
            {
                "name": node.get_id(),
                "id": nodes_index[node.get_id()],
                "value": node.get_index(CNAIndices.OUTDEGREE) + node.get_index(CNAIndices.INDEGREE)
            } 
            for node in graph.nodes()
        ],
        "links": [
            {
                "source": nodes_index[a.get_id()],
                "target": nodes_index[b.get_id()],
                "value": w
            }
            for a, b, w in graph.edges.data("weight")
        ]
    }

if __name__ == "__main__":
    folder = "pa_2019_2020"
    dr = DiacriticsRestoration()
    conv_thread = []
    for filename in os.listdir(folder):
        try:
            conv_dict = load_from_xml(os.path.join(folder, filename), diacritics_model=dr)
        except:
            print(filename)
            raise
        conv_thread.append(conv_dict)
    
    # community = Community(lang=Lang.RO, container=None, community=conv_thread, 
    #                       start_date=datetime(2019, 2, 18), end_date=datetime(2019, 5, 27))
    community = Community(lang=Lang.RO, container=None, community=conv_thread, 
                          start_date=datetime(2020, 2, 17), end_date=datetime(2020, 5, 25))
    model = get_default_model(Lang.RO)
    community.graph = CnaGraph(docs=[community], models=[model])
    
    participant_list = community.get_participants()
    names = list(map(lambda p: p.get_id(), participant_list))

    print('Participants are:')
    print(names)

    print('Begin computing indices')
    evaluate_interaction(community)
    evaluate_involvement(community)
    evaluate_textual_complexity(community)
    complete_graph = perform_sna(community)
    # with open("pa_2019-2020-graph.json", "wt", encoding="utf-8") as f:
    #     json.dump(transform_graph(complete_graph), f)
    # exit()
    compute_new_threads_metrics(community)
    
    determine_subcommunities(community, start_date=datetime(2020, 2, 17), end_date=datetime(2020, 5, 25))
    
    json_obj = {
        "words": []
    }
    all_keywords = {}
    for i, subcommunity in enumerate(community.timeframe_subcommunities):
        keywords = extract_keywords(subcommunity, max_keywords=-1, threshold=0.1)
        for value, word in keywords:
            if word not in all_keywords:
                all_keywords[word] = [0] * len(community.timeframe_subcommunities)
            all_keywords[word][i] = value
        # graph = perform_sna(subcommunity)
    keyword_scores = sorted([(word, sum(scores)) for word, scores in all_keywords], key=lambda x: x[1], reverse=True)
    keywords = sorted([word for word, score in keyword_scores[:30]])
    for word in keywords:
        json_obj["words"].append(
            {
                "activationList": [{"score": value} for value in all_keywords[word]],
                "value": word,
            }
        )
    
    # json_obj = {}
    # for participant in names:
    #     json_obj[participant] = []
    #     for index in CNAIndices:
    #         index_values = {"index": index.value}
    #         json_obj[participant].append(index_values)
    #         for i, subcommunity in enumerate(community.timeframe_subcommunities):
    #             if participant in subcommunity.participant_map:
    #                 value = subcommunity.participant_map[participant].get_index(index)
    #             else:
    #                 value = 0
    #             index_values[f"week{i+1}"] = value
    #         index_values["overall"] = community.get_participant(participant).get_index(index)

    # json_obj = {}
    # for participant in names:
    #     json_obj[participant] = {
    #         "weeks": [],
    #         "sna": {index.value: value for index, value in community.get_participant(participant).indices.items()},
    #         "complexity": community.get_participant(participant).textual_complexity_indices
    #     }
    #     for i, subcommunity in enumerate(community.timeframe_subcommunities):
    #         week = {}
    #         for index in CNAIndices:
    #             if participant in subcommunity.participant_map:
    #                 value = subcommunity.participant_map[participant].get_index(index)
    #             else:
    #                 value = 0
    #             week[index.value] = value
    #         json_obj[participant]["weeks"].append(week)
        
    with open("pa_2019-2020-keywords.json", "wt", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False)
        
	