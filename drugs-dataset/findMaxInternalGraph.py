import os, csv

class InternalGraphProcessor():

    def __init__(self):
        self.path = "internal_graphs/"

    def find_max_value(self):
        files = os.listdir(self.path)
        file_paths = [self.path + file for file in files]
        max_so_far = -1
        for file in file_paths:
            with open(file, "r") as file_descriptor:
                csv_reader = csv.reader(file_descriptor)
                for line in csv_reader:
                    for el in line:
                        max_so_far = max(max_so_far, int(el))
        return max_so_far


if __name__ == "__main__":
    processor = InternalGraphProcessor()
    print(processor.find_max_value())