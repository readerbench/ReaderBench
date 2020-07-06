import csv
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from sphinx.testing.path import path

from rb.summarization.utils.parser.custom_parser import CustomParser
from rb.summarization.systems import TextRank, Gensim, Submodular

parser = CustomParser().get_instance()

textrank_summarizer = TextRank()
gensim_summarizer = Gensim()
submodular_summarizer = Submodular()

path = Path("/home/vacioaca/Documents/summarization_input_danielle")

filenames = [f for f in listdir(path) if isfile(join(path, f))]

for filename in filenames:
    print(filename)
    df = pd.DataFrame(columns=['Text', 'Order', 'Sentence'])

    with open(path / filename, mode='r') as input_file:
        doc = input_file.read()
        sentences = parser.tokenize_sentences(doc)
        for index, sentence in enumerate(sentences):
            df = df.append({'Text': filename, 'Order': (index + 1), 'Sentence': sentence}, ignore_index=True)

    df.to_csv(path / 'csvs' / (splitext(filename)[0] + '.csv'), index=False)
    # break

# for filename in filenames:
#     print(filename)
#     df_path = path / 'csvs' / (splitext(filename)[0] + '.csv')
#     df = pd.read_csv(df_path)
#
#     with open(path / filename, mode='r') as input_file:
#         doc = input_file.read()

        # output_textrank = textrank_summarizer.summarize(doc, ratio=0.4)
        # df['textrank_0.4'] = pd.Series(output_textrank)
        # output_gensim = gensim_summarizer.summarize(doc, ratio=0.4)
        # df['gensim_0.4'] = pd.Series(output_gensim)
        # output_submodular = submodular_summarizer.summarize(doc, ratio=0.4)
        # df['submodular1_0.4'] = pd.Series(output_submodular)

    #     output_submodular = submodular_summarizer.summarize(doc, ratio=0.4)
    #     df['submodular2_0.4'] = pd.Series(output_submodular)
    #
    # df.to_csv(df_path, index=False)
    # break


# with open(path / 'summary.csv', mode='w') as csv_file:
#     fieldnames = ['Text', 'Order', 'Sentence']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for filename in filenames:
#         print(filename)
#         with open(path / filename, mode='r') as input_file:
#             doc = input_file.read()
            # output = textrank_summarizer.summarize(doc, ratio=0.25)
            # output = gensim_summarizer.summarize(doc, ratio=0.25)
            # output = submodular_summarizer.summarize(doc, ratio=0.25)
            # print(output)
            # print("\n")
        #     sentences = parser.tokenize_sentences(doc)
        #     for index, sentence in enumerate(sentences):
        #         writer.writerow({'Text': filename, 'Order': (index+1), 'Sentence': sentence})
        # break

"""
d = {
    "12_JobSearch.txt": 18,
    "15_PatientRights.txt": 17,
    "30_GlobalWarming.txt": 19,
    "20_Disabilities.txt": 20,
    "4_ChildSafety.txt": 16,
    "26_Hurricanes.txt": 18,
    "6_Smoking.txt": 12,
    "21_CheckCashing.txt": 18,
    "1_BodyScans.txt": 18,
    "8_OrganDonation.txt": 16,
    "29_Biosphere.txt": 16,
    "24_Hybrids.txt": 42,
    "11_CivilService.txt": 14,
    "2_ChildSeats.txt": 27,
    "3_HealthyBaby.txt": 13,
    "25_Google.txt": 18,
    "9_911.txt": 15,
    "7_SkinCancer.txt": 14,
    "28_CarPollution.txt": 18,
    "17_Crimes.txt": 13,
    "5_FireworksDanger.txt": 17,
    "14_ComputerVirus.txt": 17,
    "10_Internet.txt": 18,
    "23_InternetShopping.txt": 38
}


x = list(d.values())
print("\nOriginal array:")
print(x)
r1 = np.mean(x)
r2 = np.average(x)
assert np.allclose(r1, r2)
print("\nMean: ", r1)
r1 = np.std(x)
r2 = np.sqrt(np.mean((x - np.mean(x)) ** 2 ))
assert np.allclose(r1, r2)
print("\nstd: ", 1)
r1 = np.var(x)
r2 = np.mean((x - np.mean(x)) ** 2 )
assert np.allclose(r1, r2)
print("\nvariance: ", r1)
"""
