from typing import List, Dict
from datetime import datetime, timedelta

from rb.core.lang import Lang
from rb.core.cscl.community import Community
from rb.cna.cna_graph import CnaGraph
from rb.processings.cscl.participant_evaluation import perform_sna, evaluate_interaction, evaluate_involvement, evaluate_textual_complexity

from rb.core.cscl.cscl_criteria import CsclCriteria

from rb.utils.rblogger import Logger
from rb.processings.cscl.community_processing import compute_new_threads_metrics

SECONDS_IN_DAY = 60 * 60 * 24
SECONDS_IN_MONTH = SECONDS_IN_DAY * 30

def determine_subcommunities(community: Community, start_date: datetime = None, end_date: datetime = None):
    if start_date is None:
        start_date = community.first_contribution_date
    if start_date is None:
        return
    if end_date is None:
        end_date = community.last_contribution_date
    if end_date is None:
        return
    week = 1
    while start_date < end_date:
        next_date = min(start_date + timedelta(days=7), end_date)
        subcommunity = extract_subcommunity(
            parent_community=community,
            subcommunity_start=start_date,
            subcommunity_end=next_date)
        print(f"week {week}: {len(subcommunity.components)} conversations")
        week += 1
        start_date = next_date
        community.add_subcommunity(subcommunity)
        
def extract_subcommunity(parent_community: Community, subcommunity_start: datetime, subcommunity_end: datetime) -> Community:
    subcommunity = Community(lang=parent_community.lang,
                             container=parent_community,
                             community=parent_community.get_community_raw(),
                             start_date=subcommunity_start,
                             end_date=subcommunity_end)
    subcommunity.graph = CnaGraph(docs=[subcommunity], models=parent_community.graph.models)
    # evaluate_interaction(subcommunity)
    evaluate_involvement(subcommunity)
    # evaluate_textual_complexity(subcommunity)
    perform_sna(subcommunity)
    # compute_new_threads_metrics(subcommunity)
    return subcommunity
