import csv
import requests
from bs4 import BeautifulSoup
import re

class DrugBanksParser():

    def __init__(self):
        self.init_dataset_path = "ChCh-Miner_durgbank-chem-chem.csv"
        self.scrapper = DrugBankScrapper()
        self.internal_graph_mapping = {} # maps chemical elements to indexes
        self.current_mapping_limit = 0

    def parse(self):
        with open("external_graph.csv", "w") as external_graph_file:
            with open(self.init_dataset_path, 'r') as file:
                csv_reader = csv.reader(file, delimiter="\t")
                for row in csv_reader:
                    first_drug, second_drug = row[0][2:], row[1][2:]
                    
                    first_drug_smile_code = self.scrapper.get_drug_smile_code(first_drug)
                    second_drug_smile_code = self.scrapper.get_drug_smile_code(second_drug)
                    if first_drug_smile_code is not None and second_drug_smile_code is not None:
                        external_graph_file.write(first_drug + "," + second_drug + "\n")
                        self.get_and_save_internal_graph(first_drug, first_drug_smile_code)
                        self.get_and_save_internal_graph(second_drug, second_drug_smile_code)
        self.write_dataset_summary()

    # Example smile rep: CCCC1=NC(C)=C2N1NC(=NC2=O)C1=C(OCC)C=CC(=C1)S(=O)(=O)N1CCN(CC)CC1
    def get_and_save_internal_graph(self, drug_code, smile_repr):
        substituted_smile_repr = re.sub(r'\W+', '', smile_repr)
        smile_repr_unwrapped = ""
        previous_char = ""
        for idx in range(0, len(smile_repr)):
            if smile_repr[idx].isalpha(): 
                previous_char = smile_repr[idx]
                smile_repr_unwrapped += previous_char
            if smile_repr[idx].isdigit(): 
                smile_repr_unwrapped += (int(smile_repr[idx])-1)*previous_char

        with open("internal_graphs/" + drug_code + ".csv", "w") as file:
            for idx in range(0, len(smile_repr_unwrapped)):
                if smile_repr_unwrapped[idx] not in self.internal_graph_mapping:
                    self.internal_graph_mapping[smile_repr_unwrapped[idx]] = self.current_mapping_limit
                    self.current_mapping_limit += 1
                line = str(self.internal_graph_mapping[smile_repr_unwrapped[idx]]) + ","
                for x in range(-1, 2, 2): # -1, 1
                    if idx+x >= 0 and idx+x < len(smile_repr_unwrapped):
                        char = smile_repr_unwrapped[idx+x]
                        if char not in self.internal_graph_mapping:
                            self.internal_graph_mapping[char] = self.current_mapping_limit
                            self.current_mapping_limit += 1
                        line += str(self.internal_graph_mapping[char])
                        line += ","
                line += "\n"
                file.write(line)
        return smile_repr

    def write_dataset_summary(self):
        with open("summary.csv", "w") as file:
            file.write("Number of internal nodes types " + str(self.current_mapping_limit))

class DrugBankScrapper():
    def __init__(self):
        pass

    def get_drug_smile_code(self, drug_id):
        url = "https://www.drugbank.ca/drugs/DB" + drug_id
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        all_smiles = soup.find_all(string="SMILES")
        for element in all_smiles[1:]:
            for next_el in list(element.next_elements)[:1]:
                smile_repr = next_el.string
                return smile_repr

if __name__ == "__main__":
    DrugBanksParser = DrugBanksParser()
    DrugBanksParser.parse()