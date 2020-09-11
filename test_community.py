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

if __name__ == "__main__":
    folder = "pa_2018_2019"
    conv_thread = []
    for filename in os.listdir(folder):
        try:
            conv_dict = load_from_xml(os.path.join(folder, filename))
        except:
            print(filename)
            raise
        conv_thread.append(conv_dict)
    community = Community(lang=Lang.RO, container=None, community=conv_thread, 
                          start_date=datetime(2019, 2, 18), end_date=datetime(2019, 5, 27))
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
    perform_sna(community)
    compute_new_threads_metrics(community)
    
    determine_subcommunities(community, start_date=datetime(2019, 2, 18), end_date=datetime(2019, 5, 27))
    
    json_obj = {}
    for participant in names:
        json_obj[participant] = []
        for index in CNAIndices:
            index_values = {"index": index.value}
            json_obj[participant].append(index_values)
            for i, subcommunity in enumerate(community.timeframe_subcommunities):
                if participant in subcommunity.participant_map:
                    value = subcommunity.participant_map[participant].get_index(index)
                else:
                    value = 0
                index_values[f"week{i+1}"] = value
            index_values["overall"] = community.get_participant(participant).get_index(index)
    with open("pa_2018-2019.json", "wt", encoding="utf-8") as f:
        json.dump(json_obj, f)
        
	