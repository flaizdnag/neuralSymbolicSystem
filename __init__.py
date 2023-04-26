import json
from os import listdir
import numpy as np
from os.path import isfile, join
import src

import src.neural_network.network as network
import nn_lp_jsons_main.jsons as jsons


def get_nn_recipe(lp_dict: dict) -> dict:
    lp = src.logic.LogicProgram.from_dict(lp_dict['lp'])
    ag = src.logic.Clause.from_dict(lp_dict['abductive_goal'])

    factors_d = lp_dict['factors']
    factors = src.logic.Factors.from_dict(factors_d)

    return src.connect.get_nn_recipe(lp, ag, factors)


def loop(recipe: str) -> dict:
    data = dict()

    with open(recipe, 'r') as json_file:
        experiment_recipe = json.load(json_file)

    data['lp_before'] = experiment_recipe['lp']
    data['lp_before_params'] = jsons.lp_params(experiment_recipe)
    data['neural_network_factors'] = experiment_recipe['factors']

    nn_recipe = get_nn_recipe(experiment_recipe)

    data['nn_recipe'] = nn_recipe
    # data['nn_before_params'] = jsons.nn_params(nn_recipe)

    nn = network.NeuralNetwork3L.from_dict(nn_recipe)

    data['nn_before'] = nn._pack()

    nn.train(
        np.array(experiment_recipe['training_data']['inputs']),
        np.array(experiment_recipe['training_data']['outputs']),
        experiment_recipe['training_data']['epochs'],
        on_stabilised=experiment_recipe['training_data']['epochs'],
        stop_when=lambda e:
            e <= experiment_recipe['training_data']['stop_error']
    )

    data['nn_after'] = nn._pack()
    figure = nn.draw()

    data['errors'] = nn.errors

    data['io_pairs'] = nn.get_io_pairs()

    data['lp_after'] = nn.to_lp()
    data['lp_after_params'] = jsons.lp_params(experiment_recipe)

    return (data, figure)


def run_loop():
    """
    Orchestrates the calculations and writing json files.
    """
    experiments_path = 'experiment_versions/'
    experiment_versions = [f for
                           f in listdir(experiments_path)
                           if isfile(join(experiments_path, f))
                           and f[0] != "."]

    print(f"experiment versions: {experiment_versions}")

    for file in experiment_versions:
        # TODO change this loop: it should be regulated by eperiment parameter
        file_name_no_ext = file.split('.')[0]
        for i in range(10):
            print(f"example {i+1} from experiment {file_name_no_ext}")
            (result, figure) = loop(join(experiments_path, file))
            json_name = f"results/{file_name_no_ext}_{str(i).zfill(5)}.json"
            figure_name = f"results/{file_name_no_ext}_{str(i).zfill(5)}.png"
            figure.savefig(figure_name)
            with open(json_name, 'w') as json_file:
                json.dump(result, json_file)


if __name__ == '__main__':
    run_loop()
