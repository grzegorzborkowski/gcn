import csv
import requests
from bs4 import BeautifulSoup

class DrugBanksParser():

    def __init__(self):
        self.init_dataset_path = "ChCh-Miner_durgbank-chem-chem.csv"
        self.scrapper = DrugBankScrapper()
        self.internal_graph_mapping = {} # maps chemical elements to indexes

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

    # TODO: implement
    def get_and_save_internal_graph(self, drug_code, smile_repr):
        print ("smile_rep" + smile_repr)
        return smile_repr

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