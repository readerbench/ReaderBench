from typing import List, Dict

from rb.core.lang import Lang
from rb.core.cscl.community import Community
from rb.processings.cscl.community_processing import CommunityProcessing
from rb.core.cscl.cscl_criteria import CsclCriteria

from rb.utils.rblogger import Logger

SECONDS_IN_DAY = 60 * 60 * 24
SECONDS_IN_MONTH = SECONDS_IN_DAY * 30

class CommunityTimeProcessing:

    def determine_subcommunities(self, community: Community, month_increment: int, day_increment: int):
        if (community.get_first_contribution_date() is None):
            return

        subcommunity_start = community.get_first_contribution_date()

        while True:
            # todo take into account length of month and current date
            subcommunity_end = subcommunity_start

            subcommunity_end += (month_increment * SECONDS_IN_MONTH)
            subcommunity_end += (day_increment * SECONDS_IN_DAY)

            if subcommunity_end < community.get_last_contribution_date():
                subcommunity = self.extract_subcommunity(parent_community=community,
                                                        subcommunity_start=subcommunity_start,
                                                        subcommunity_end=subcommunity_end)

                community.add_subcommunity(subcommunity)

                subcommunity_start = subcommunity_end
            else:
                break

        if subcommunity_start < community.get_last_contribution_date():
            subcommunity = self.extract_subcommunity(parent_community=community,
                                                    subcommunity_start=subcommunity_start,
                                                    subcommunity_end=community.get_last_contribution_date())

            community.add_subcommunity(subcommunity)
        

    def extract_subcommunity(self, parent_community: Community, subcommunity_start: int, subcommunity_end: int) -> Community:
        subcommunity = Community(lang=parent_community.lang,
                                community=parent_community.get_community_raw(),
                                start_date=subcommunity_start,
                                end_date=subcommunity_end)

        community_processing = CommunityProcessing()
        
        community_processing.determine_participant_contributions(subcommunity)
        community_processing.determine_participation(subcommunity)
        community_processing.compute_sna_metrics(subcommunity)

        return subcommunity

    def model_time_evoluton(self, community: Community):
        return