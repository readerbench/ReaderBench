from typing import List, Dict

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.block import Block
from rb.core.pos import POS
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.cscl_indices import CsclIndices

from rb.utils.rblogger import Logger

def get_block_importance(block_importance: Dict[Block, Dict[Block, float]], a: Block, b: Block) -> float:
	if not (a in block_importance):
		return 0
	if not (b in block_importance[a]):
		return 0

	return block_importance[a][b]

def determine_participant_contributions(community: Community):
    for conversation in community.get_conversations():
        for p in conversation.get_participants():
            contributions = conversation.get_participant_contributions(p.get_id())

            for c in contributions:
                if community.is_eligible(c.get_timestamp()):
                    p.add_eligible_contribution(c)

                    if c.is_significant():
                        p.add_significant_contribution(c)

                    current_value = p.get_index(CsclIndices.NO_CONTRIBUTION)
                    p.set_index(CsclIndices.NO_CONTRIBUTION, current_value + 1)

                    for word in c.get_words():
                        if word.pos.to_wordnet() == 'n':
                            current_value = p.get_index(CsclIndices.NO_NOUNS)
                            p.set_index(CsclIndices.NO_NOUNS, current_value + 1)
                        if word.pos.to_wordnet() == 'v':
                            current_value = p.get_index(CsclIndices.NO_VERBS)
                            p.set_index(CsclIndices.NO_VERBS, current_value + 1)


def determine_participation(community: Community):
    cna_graph = community.graph
    importance = cna_graph.importance
    block_importance = cna_graph.block_importance

    community.init_scores()

    for conversation in community.get_conversations():
        contributions = conversation.get_contributions()

        for i, contribution1 in enumerate(contributions):
            if community.is_eligible(contribution1.get_timestamp()):
                p1 = contribution1.get_participant().get_id()
                participant = contribution1.get_participant()

                community.update_score(p1, p1, importance[contribution1])

                current_value = participant.get_index(CsclIndices.SCORE)
                participant.set_index(CsclIndices.SCORE, current_value + importance[contribution1])

                current_value = participant.get_index(CsclIndices.SOCIAL_KB)
                parent_contribution = contribution1.get_parent()

                if parent_contribution != None:
                    current_value += (get_block_importance(block_importance, contribution1, parent_contribution) *
                                        importance[contribution1])

                    participant.set_index(CsclIndices.SOCIAL_KB, current_value)

                for j in range(0, i):
                    contribution2 = contributions[j]

                    if get_block_importance(block_importance, contribution1, contribution2) > 0:
                        p2 = contribution2.get_participant().get_id()

                        added_kb = importance[contribution1] * get_block_importance(block_importance,
                                                                                        contribution1, contribution2)

                        community.update_score(p1, p2, added_kb)


def perform_sna(community: Community, needs_anonymization: bool):
    participants = community.get_participants()

    for i, p1 in enumerate(participants):
        for j, p2 in enumerate(participants):
            if i != j:
                current_value = p1.get_index(CsclIndices.OUTDEGREE)
                p1.set_index(CsclIndices.OUTDEGREE, current_value +
                                            community.get_score(p1.get_id(), p2.get_id()))

                current_value = p2.get_index(CsclIndices.INDEGREE)
                p2.set_index(CsclIndices.INDEGREE, current_value +
                                            community.get_score(p1.get_id(), p2.get_id()))
            else:
                current_value = p1.get_index(CsclIndices.OUTDEGREE)
                p1.set_index(CsclIndices.OUTDEGREE, current_value +
                                            community.get_score(p1.get_id(), p1.get_id()))

def compute_sna_metrics(community: Community):
    cna_graph = community.graph
    importance = cna_graph.importance

    perform_sna(community, True)

    for conversation in community.get_conversations():
        for contribution in conversation.get_contributions():
            p = contribution.get_participant()

            current_value = p.get_index(CsclIndices.NO_NEW_THREADS)
            p.set_index(CsclIndices.NO_NEW_THREADS, current_value + 1)

            current_value = p.get_index(CsclIndices.NEW_THREADS_OVERALL_SCORE)
            p.set_index(CsclIndices.NEW_THREADS_OVERALL_SCORE, current_value + importance[conversation])

            current_value = p.get_index(CsclIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB)
            total_kb = conversation.get_cumulative_social_kb()
            p.set_index(CsclIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB, current_value + total_kb)

            # todo add voice pmi evolution

            current_value = p.get_index(CsclIndices.AVERAGE_LENGTH_NEW_THREADS)
            p.set_index(CsclIndices.AVERAGE_LENGTH_NEW_THREADS, current_value + len(contribution.text))

            break
    
    for p in community.get_participants():
        new_threads = p.get_index(CsclIndices.NO_NEW_THREADS)
        total_length = p.get_index(CsclIndices.AVERAGE_LENGTH_NEW_THREADS)

        if new_threads > 0:
            average_length = total_length / new_threads
            p.set_index(CsclIndices.AVERAGE_LENGTH_NEW_THREADS, average_length)