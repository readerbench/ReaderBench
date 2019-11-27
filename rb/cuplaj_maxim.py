from munkres import Munkres, DISALLOWED, print_matrix

def get_maximum_coupling(author1_details, author2_details, sim_dict):
    author1_articles = author1_details[1]
    author2_articles = author2_details[1]
    # size = max(len(author1_articles), len(author2_articles))

    m = [[DISALLOWED for x in range(len(author2_articles))] for y in range(len(author1_articles))]

    for i1, art1 in enumerate(author1_articles):
        for i2, art2 in enumerate(author2_articles):
            if (art1, art2) in sim_dict:
                if sim_dict[(art1, art2)] > 0.3:
                    m[i1][i2] = (1 - sim_dict[(art1, art2)]) * 100
            elif (art2, art1) in sim_dict:
                if sim_dict[(art2, art1)] > 0.3:
                    m[i1][i2] = (1 - sim_dict[(art2, art1)]) * 100


    mumu = Munkres()
    indexes = mumu.compute(m)
    total = 0
    for row, column in indexes:
        value = 1 - (m[row][column] / 100)
        print(row, column)
        total += value
    print(total)
    return total

if __name__ == "__main__":
    # a1 = ("Mihai Dascalu", ["A1", "A2", "A3"])
    # a2 = ("Profesorul Xavier", ["B1", "B2", "B3", "B4"])
    # s_dict = {
    #     ("A1", "B1"): 0.5, ("A1", "B2"): 0.2, ("A1", "B3"): 0.7, ("A1", "B4"): 0.87,
    #     ("A2", "B1"): 0.25, ("A2", "B2"): 0.49, ("A2", "B3"): 0.81, ("A2", "B4"): 0.33,
    #     ("A3", "B1"): 0.33, ("A3", "B2"): 0.44, ("A3", "B3"): 0.45, ("A3", "B4"): 0.87,
    # }

    a1 = ("Mihai Dascalu", ["A1", "A2"])
    a2 = ("Profesorul Xavier", ["B1", "B2", "B3", "B4"])
    s_dict = {
        ("A1", "B1"): 0.4, ("A1", "B2"): 0.2, ("A1", "B3"): 0.6, ("A1", "B4"): 0.1,
        ("A2", "B1"): 0.5, ("A2", "B2"): 0.1, ("A2", "B3"): 0.3, ("A2", "B4"): 0.7,
    }

    get_maximum_coupling(a1, a2, s_dict)