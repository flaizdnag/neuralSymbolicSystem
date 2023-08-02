import json
import os

# list of parameters
ahln = [5.0, 2.0, 1.0, 0.0]
bias = [-6.0, -1.0, 0.0, 1.0, 6.0]
problem =                   # number of the problem
y =                         # training set number
output_path = f"./y={y}"    #path to separate folder


def create_json_file(path, filename, data):
    file_path = os.path.join(path, filename)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


def generate_json_data():
    for i in ahln:
        for j in bias:
            filename = f"p{problem}_{j:.0f}_{i:.0f}_{y}.json"
            # atoms added manually
            json_data = {
                "lp": {
                    "facts": [],
                    "assumptions": [],
                    "clauses": [
                        {
                            "tag": "Cl",
                            "clHead": {"idx": , "label": ""},
                            "clPAtoms": [
                                {"idx": 1, "label": ""},
                            ],
                            "clNAtoms": [],
                        },
                        {
                            "tag": "Cl",
                            "clHead": {"idx": , "label": ""},
                            "clPAtoms": [{"idx": , "label": ""}],
                            "clNAtoms": [],
                        },
                        {
                            "tag": "Cl",
                            "clHead": {"idx": , "label": ""},
                            "clPAtoms": [{"idx": , "label": ""}],
                            "clNAtoms": [],
                        },
                        {
                            "tag": "Cl",
                            "clHead": {"idx": , "label": ""},
                            "clPAtoms": [{"idx": , "label": ""}],
                            "clNAtoms": [],
                        },
                        {
                            "tag": "Cl",
                            "clHead": {"idx": , "label": ""},
                            "clPAtoms": [{"idx": , "label": ""}],
                            "clNAtoms": [],
                        },
                    ],
                },
                "abductive_goal": {
                    "tag": "Cl",
                    "clHead": {"idx": , "label": ""},
                    "clPAtoms": [],
                    "clNAtoms": [],
                },
                "factors": {
                    "beta_for_activation_function": 1.0,
                    "number_of_additional_neurons": i,
                    "additional_weights_range": 0.01,
                    "bias_for_additional_neurons": j,
                    "w_factor": 0.1,
                    "amin_factor": 0.1,
                },
                "training_data": {
                    "inputs": [[-1, -1, -1, -1, -1, -1, -1]],
                    "outputs": [[1, 1, 1, 1, -1, 1]],       #outputs changed manually 
                    "epochs": 10000,
                    "on_stabilised": 1,
                    "stop_error": 0.02,
                },
            }
            create_json_file(output_path, filename, json_data)
            print(f"JSON file '{filename}' has been created successfully!")
    

generate_json_data()