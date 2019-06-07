from rb.core.lang import Lang
from rb.utils.downloader import download_aoa

import csv
from copy import deepcopy

from typing import Dict


class AgeOfAcquisition():

    def __init__(self, lang: Lang):
        self.lang = lang
        download_aoa(lang)
        self.aoa_dict = self.read_aoa_file()


    """
    Function that populates a dictionary with data from the AoA file for a specific language.
    Returns a dictionary that has as the key the word and as the value another dict
        representing the type of AoA (example: Kuperman) and its value
    """
    def read_aoa_file(self) -> Dict[str, Dict[str, float]]:
        aoa_dict = {}
        headers = []
        with open('resources/{}/aoa/AoA.csv'.format(self.lang.value)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for index, row in enumerate(csv_reader):
                if index == 0: # delimitator -> ignore
                    continue
                if index == 1:
                    headers = deepcopy(row)
                    continue
                # else
                for i, elem in enumerate(row):
                    if i == 0:
                        aoa_dict[elem] = {}
                    else:
                        aoa_dict[row[0]][headers[i]] = float(elem) if elem else None

        return aoa_dict


    def get_bird_value(self, word: str) -> float:
        if word not in self.aoa_dict or "Bird" not in self.aoa_dict[word]:
            return None
        return self.aoa_dict[word]["Bird"]


    def get_bristol_value(self, word: str) -> float:
        if word not in self.aoa_dict or "Bristol" not in self.aoa_dict[word]:
            return None
        return self.aoa_dict[word]["Bristol"]


    def get_cortese_value(self, word: str) -> float:
        if word not in self.aoa_dict or "Cortese" not in self.aoa_dict[word]:
            return None
        return self.aoa_dict[word]["Cortese"]


    def get_kuperman_value(self, word: str) -> float:
        if word not in self.aoa_dict or "Kuperman" not in self.aoa_dict[word]:
            return None
        return self.aoa_dict[word]["Kuperman"]

    
    def get_shock_value(self, word: str) -> float:
        if word not in self.aoa_dict or "Shock" not in self.aoa_dict[word]:
            return None
        return self.aoa_dict[word]["Shock"]