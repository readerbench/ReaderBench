from typing import Dict, List, Union

import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality, eigenvector_centrality
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.block import Block
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.utils.rblogger import Logger


def get_block_importance(graph: nx.DiGraph, a: Block, b: Block) -> float:
    return graph.edges[a, b]["weight"] if (a, b) in graph.edges else 0.


def evaluate_interaction(conversation: Union[Conversation, Community]):
    cna_graph = conversation.container.graph if conversation.container is not None else conversation.graph
    
    participants = conversation.get_participants()
    if isinstance(conversation, Community):
        contributions = [contribution for conv in conversation.components for contribution in conv.get_contributions()]
    else:
        contributions = conversation.get_contributions()
    conversation.init_scores()

    if (len(participants) == 0):
        return

    for i, contribution1 in enumerate(contributions):
        p1 = contribution1.get_participant().get_id()

        for contribution2 in contributions[:i]:
            weight = get_block_importance(cna_graph.filtered_graph, contribution1, contribution2)
            if weight > 0:
                p2 = contribution2.get_participant().get_id()
                conversation.update_score(p1, p2, weight)


def evaluate_involvement(conversation: Union[Conversation, Community]):
    cna_graph = conversation.container.graph if conversation.container is not None else conversation.graph
    participants = conversation.get_participants()

    if (len(participants) == 0):
        return
    if isinstance(conversation, Community):
        conversations = conversation.components
    else:
        conversations = [conversation]
    for conv in conversations:
        for i, contribution in enumerate(conv.get_contributions()):
            p = conversation.participant_map[contribution.get_participant().get_id()]

            current_value = p.get_index(CNAIndices.CONTRIBUTIONS_SCORE)
            p.set_index(CNAIndices.CONTRIBUTIONS_SCORE, current_value + cna_graph.importance[contribution])
            original_conv = conversation.container.get_conversation(conv.conv_id) if conversation.container else conv
            added_value = 0
            for prev in original_conv.get_contributions():
                if contribution.index == prev.index:
                    break
                if p != prev.get_participant():
                    added_value += get_block_importance(cna_graph.filtered_graph, contribution, prev)
            p.set_index(CNAIndices.SOCIAL_KB, p.get_index(CNAIndices.SOCIAL_KB) + added_value)

def evaluate_textual_complexity(conversation: Conversation):
    participants = conversation.get_participants()
    
    for p in participants:
        contributions = conversation.get_participant_contributions(p.get_id())
        p.set_eligible_contributions(contributions)

        Conversation.create_participant_conversation(conversation.lang, p)

        own_conversation = p.own_conversation

        print('Begin computing textual complexity')
        compute_indices(doc=own_conversation)
        print('Finished computing textual complexity')

        for key, value in own_conversation.indices.items():
            p.set_textual_index(repr(key), value)


def perform_sna(conversation: Union[Conversation, Community], needs_anonymization = False) -> nx.DiGraph:
    participants = conversation.get_participants()
    cna_graph = conversation.graph
    participants_graph = nx.DiGraph()
    participants_graph.add_nodes_from(participants)
    conversations = conversation.get_conversations() if isinstance(conversation, Community) else [conversation]
    for conv in conversations:
        for i, u in enumerate(conv.get_contributions()):
            for v in conv.get_contributions()[(i+1):]:
                if u.get_participant() != v.get_participant() and u.timestamp < v.timestamp:
                    edge = cna_graph.filtered_graph.get_edge_data(u, v)
                    if edge is None:
                        continue
                    weight = edge["weight"]
                    if participants_graph.has_edge(u.get_participant(), v.get_participant()):
                        participants_graph[u.get_participant()][v.get_participant()]["weight"] += weight
                    else:
                        participants_graph.add_edge(u.get_participant(), v.get_participant(), weight=weight)
    betweeness = betweenness_centrality(participants_graph)
    closeness = closeness_centrality(participants_graph)
    eigen = eigenvector_centrality(participants_graph, max_iter=1000, tol=1.0e-5)
    for p in participants:
        p.set_index(CNAIndices.OUTDEGREE, participants_graph.out_degree(p, weight="weight"))
        p.set_index(CNAIndices.INDEGREE, participants_graph.in_degree(p, weight="weight"))
        p.set_index(CNAIndices.BETWEENNESS, betweeness[p])
        p.set_index(CNAIndices.CLOSENESS, closeness[p])
        p.set_index(CNAIndices.EIGENVECTOR, eigen[p])
    return participants_graph
