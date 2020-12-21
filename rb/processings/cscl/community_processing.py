from typing import Dict, List

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.block import Block
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.community import Community
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.processings.cscl.participant_evaluation import get_block_importance, perform_sna
from rb.utils.rblogger import Logger

def compute_new_threads_metrics(community: Community):
    cna_graph = community.graph
    importance = cna_graph.importance
    
    for conversation in community.get_conversations():
        participant = conversation.get_contributions()[0].get_participant()
        p = community.get_participant(participant.get_id())
        current_value = p.get_index(CNAIndices.NO_NEW_THREADS)
        p.set_index(CNAIndices.NO_NEW_THREADS, current_value + 1)
        for contribution in conversation.get_contributions():
            participant = contribution.get_participant()
            p = community.get_participant(participant.get_id())

            current_value = p.get_index(CNAIndices.NEW_THREADS_OVERALL_SCORE)
            p.set_index(CNAIndices.NEW_THREADS_OVERALL_SCORE,
                        current_value + importance[conversation])

            # current_value = p.get_index(
            #     CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB)
            # TODO
            # total_kb = conversation.get_cumulative_social_kb()
            # p.set_index(CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB,
            #             current_value + total_kb)

            # todo add voice pmi evolution
            current_value = p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS)
            p.set_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS,
                        current_value + 1)
    for p in community.get_participants():
        new_threads = p.get_index(CNAIndices.NO_NEW_THREADS)
        total_length = p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS)

        if new_threads > 0:
            average_length = total_length / new_threads
            p.set_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS, average_length)