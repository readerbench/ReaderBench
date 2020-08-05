from typing import Dict, List

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.block import Block
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.community import Community
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.processings.cscl.participant_evaluation import get_block_importance
from rb.utils.rblogger import Logger

def determine_participant_contributions(community: Community):
    for conversation in community.get_conversations():
        for participant in conversation.get_participants():
            p = community.get_participant(participant.get_id())
            contributions = conversation.get_participant_contributions(p.get_id())

            for c in contributions:
                if community.is_eligible(c.get_timestamp()):
                    p.add_eligible_contribution(c)

                    if c.is_significant():
                        p.add_significant_contribution(c)
                current_value = p.get_index(CNAIndices.NO_CONTRIBUTION)
                p.set_index(CNAIndices.NO_CONTRIBUTION, current_value + 1)
    
# must be called after determine participant_contributions
def determine_textual_complexity(community: Community):
    for p in community.get_participants():
        Conversation.create_participant_conversation(community.lang, p)
        conversation = p.own_conversation

        print('Begin computing textual complexity')
        compute_indices(doc=conversation)
        print('Finished computing textual complexity')

        for key, value in conversation.indices.items():
            p.set_textual_index(repr(key), value)


def determine_participation(community: Community):
    cna_graph = community.graph
    community.init_scores()

    for conversation in community.get_conversations():
        contributions = conversation.get_contributions()

        for i, contribution1 in enumerate(contributions):
            if community.is_eligible(contribution1.get_timestamp()):
                p1 = contribution1.get_participant().get_id()
                participant = community.get_participant(p1)

                community.update_score(p1, p1, cna_graph.importance[contribution1])

                current_value = participant.get_index(CsclIndices.CONTRIBUTIONS_SCORE)
                participant.set_index(
                    CsclIndices.CONTRIBUTIONS_SCORE, current_value + cna_graph.importance[contribution1])

                current_value = participant.get_index(CsclIndices.SOCIAL_KB)
                parent_contribution = contribution1.get_parent()

                if parent_contribution != None:
                    current_value += get_block_importance(cna_graph.filtered_graph, contribution1, parent_contribution)
                    participant.set_index(CsclIndices.SOCIAL_KB, current_value)

                for j in range(0, i):
                    contribution2 = contributions[j]
                    weight = get_block_importance(cna_graph.filtered_graph, contribution1, contribution2)
                    if weight > 0:
                        p2 = contribution2.get_participant().get_id()
                        community.update_score(p1, p2, weight)



def perform_sna(community: Community, needs_anonymization: bool):
    participants = community.get_participants()

    for i, p1 in enumerate(participants):
        for j, p2 in enumerate(participants):
            if i != j:
                current_value = p1.get_index(CNAIndices.OUTDEGREE)
                p1.set_index(CNAIndices.OUTDEGREE, current_value +
                             community.get_score(p1.get_id(), p2.get_id()))
                current_value = p2.get_index(CNAIndices.INDEGREE)
                p2.set_index(CNAIndices.INDEGREE, current_value +
                             community.get_score(p1.get_id(), p2.get_id()))
            else:
                current_value = p1.get_index(CNAIndices.OUTDEGREE)
                p1.set_index(CNAIndices.OUTDEGREE, current_value +
                             community.get_score(p1.get_id(), p1.get_id()))


def compute_sna_metrics(community: Community):
    cna_graph = community.graph
    importance = cna_graph.importance
    perform_sna(community, True)

    for conversation in community.get_conversations():
        for contribution in conversation.get_contributions():
            participant = contribution.get_participant()
            p = community.get_participant(participant.get_id())

            current_value = p.get_index(CNAIndices.NO_NEW_THREADS)
            p.set_index(CNAIndices.NO_NEW_THREADS, current_value + 1)

            current_value = p.get_index(CNAIndices.NEW_THREADS_OVERALL_SCORE)
            p.set_index(CNAIndices.NEW_THREADS_OVERALL_SCORE,
                        current_value + importance[conversation])

            current_value = p.get_index(
                CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB)
            # TODO
            # total_kb = conversation.get_cumulative_social_kb()
            # p.set_index(CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB,
            #             current_value + total_kb)

            # todo add voice pmi evolution
            current_value = p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS)
            p.set_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS,
                        current_value + len(contribution.text))
            break
    for p in community.get_participants():
        new_threads = p.get_index(CNAIndices.NO_NEW_THREADS)
        total_length = p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS)

        if new_threads > 0:
            average_length = total_length / new_threads
            p.set_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS, average_length)
