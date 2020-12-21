from typing import List, Dict

from rb.core.block import Block
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.community import Community
from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.regularity_indices import RegularityIndices

from rb.utils.rblogger import Logger
from bisect import bisect_left
from copy import deepcopy
import math
from scipy.special import entr
from numpy.linalg import norm
import numpy as np

L_d = None
L_m = None
L_w = None
user_list_map = None


def get_all_timestamps_for_user(given_user, community: Community):
    timestamps = []
    user_contribution = community.get_participant_contributions(given_user)
    for contribution in user_contribution:
        timestamp = contribution.get_timestamp()
        if timestamp not in timestamps:
            timestamps.append(timestamp)

    timestamps.sort()
    return timestamps


def get_user_list_map(community: Community):
    users = community.get_participants()
    for user in users:
        timestamps = get_all_timestamps_for_user(user, community)
        user_list_map[user] = timestamps


def F(user, W, x):
    user_list = user_list_map[user]
    to_search = W * x
    index = bisect_left(user_list, to_search)

    if index == len(user_list):
        return 0
    elif user_list[index] <= (W * x + W):
        return 1

    return 0


def D(user, h):
    total = 0

    for i in range(L_d):
        total += F(user, 60, 24 * i + h)

    return total


def W(user, d):
    total = 0

    for i in range(L_w):
        total += F(user, 60 * 24, 7 * i + d)

    return total


def compute_time_measures(user):
    user_d = []
    user_w = []

    for h in range(24):
        user_d.append(D(user, h))
    for d in range(7):
        user_w.append(W(user, d))

    max_d = max(user_d)
    max_w = max(user_w)
    total_d = sum(user_d)
    total_w = sum(user_w)
    normalised_d = deepcopy(user_d)
    normalised_w = deepcopy(user_w)

    if total_d != 0:
        normalised_d = list(map(lambda x: float(x) / total_d, user_d))
    if total_w != 0:
        normalised_w = list(map(lambda x: float(x) / total_w, user_w))

    e_d = 0
    for h in range(24):
        if normalised_d[h] != 0:
            e_d += (normalised_d[h] * math.log(normalised_d[h]))

    e_w = 0
    for d in range(7):
        if normalised_w[d] != 0:
            e_w += (normalised_w[d] * math.log(normalised_w[d]))

    PDH = (math.log(24) - e_d) * max_d
    PWD = (math.log(7) - e_w) * max_w

    return PDH, PWD


def P(user, d, k):
    total = 0

    for i in range(24):
        total += F(user, 60, 24 * (d + 7 * k) + i)

    return total


def get_profile(user, k):
    return [P(user, d, k) for d in range(7)]


def active(user, k):
    profile = get_profile(user, k)
    active_days = []

    for d in range(7):
        if profile[d] != 0:
            active_days.append(d)

    return active_days


def JSD(P, Q):
    _P, _Q = np.array(P), np.array(Q)
    sum_p = sum(P)
    sum_q = sum(Q)

    if sum_p != 0:
        _P = _P / sum_p
    if sum_q != 0:
        _Q = _Q / sum_q

    return entr(0.5 * (_P + _Q)).sum() - 0.5 * (entr(_P).sum() + entr(_Q).sum())


def similarity_1(user, i, j):
    active_i = active(user, i)
    active_j = active(user, j)
    active_common = list(set(active_i) & set(active_j))

    max_len = max(len(active_i), len(active_j))
    if max_len == 0:
        return 0

    return float(len(active_common)) / max_len


def similarity_2(user, i, j):
    profile_i = get_profile(user, i)
    profile_j = get_profile(user, j)

    return 1 - (JSD(profile_i, profile_j) / math.log(2))


def similarity_3(user, i, j):
    active_i = active(user, i)
    active_j = active(user, j)
    active_all = list(set(active_i) | set(active_j))

    if len(active_all) == 0:
        return 0

    total = 0
    for d in range(7):
        p_i = P(user, d, i)
        p_j = P(user, d, j)

        if (p_i + p_j) != 0:
            total += math.pow(float(p_i - p_j) / (p_i + p_j), 2)

    return 1 - float(total) / len(active_all)


def compute_similarities(user):
    ws1_total = 0
    ws2_total = 0
    ws3_total = 0
    count = 0

    for i in range(L_w):
        for j in range(i + 1, L_w):
            count += 1
            ws1_total += similarity_1(user, i, j)
            ws2_total += similarity_2(user, i, j)
            ws3_total += similarity_3(user, i, j)

    if count == 0:
        count = 1

    return float(ws1_total) / count, float(ws2_total) / count, float(ws3_total) / count


def determine_regularity(community: Community):
    global L_d
    global L_m
    global L_w
    users = community.get_participants()
    get_user_list_map(community)
    for i in range(0, len(users)):
        user_timestamps = user_list_map[users[i]]
        last_comment_timestamp = user_timestamps[len(user_timestamps) - 1]
        first_comment_timestamp = user_timestamps[0]

        difference = last_comment_timestamp - first_comment_timestamp
        L_d = int(float(difference) / (60 * 60 * 24))
        L_m = int(float(difference) / (60))
        L_w = int(float(difference) / (60 * 60 * 24 * 7))

        user_list = list(map(lambda t: int((t - first_comment_timestamp) / 60), user_timestamps))
        user_list.sort()

        user_list_map[users[i]] = user_list

        PDH, PWD = compute_time_measures(users[i])
        WS1, WS2, WS3 = compute_similarities(users[i])

        users[i].set_index(RegularityIndices.PDH, str(round(PDH, 2)))
        users[i].set_index(RegularityIndices.PWD, str(round(PWD, 2)))
        users[i].set_index(RegularityIndices.WS1, str(round(WS1, 2)))
        users[i].set_index(RegularityIndices.WS2, str(round(WS2, 2)))
        users[i].set_index(RegularityIndices.WS3, str(round(WS3, 3)))
