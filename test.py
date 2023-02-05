import src
import json
import numpy as np

import src.neural_network.network as network
import nn_lp_jsons_main.jsons as jsons


def get_nn_recipe(lp_dict: dict, param: dict) -> dict:
    lp = src.logic.LogicProgram.from_dict(lp_dict['lp'])
    ag = src.logic.Clause.from_dict(lp_dict['abductive_goal'])

    factors_d = lp_dict['factors']
    factors_d.update(param)
    factors = src.logic.Factors.from_dict(factors_d)

    return src.connect.get_nn_recipe(lp, ag, factors)


def loop(recipe: str, param: dict) -> dict:
    data = dict()

    with open(recipe, 'r') as json_file:
        lp1 = json.load(json_file)

    data['lp_before'] = lp1['lp']
    data['lp_before_params'] = jsons.lp_params(lp1)
    data['neural_network_factors'] = lp1['factors']

    nn_recipe = get_nn_recipe(lp1, param)

    data['nn_recipe'] = nn_recipe
    data['nn_before_params'] = jsons.nn_params(nn_recipe)

    nn = network.NeuralNetwork3L.from_dict(nn_recipe)

    print('positive output:', nn.get_lp()['positive_out'])
    print('negative output:', nn.get_lp()['negative_out'])

    data['nn_before'] = nn._pack()

    nn.train(
        np.array([[-1, -1, -1, -1]]),
        np.array([[1, 1, 1]]),
        10000,
        on_stabilised=True,
        stop_when=lambda e: e <= 0.02
    )

    data['nn_after'] = nn._pack()

    data['errors'] = nn.errors

    data['io_pairs'] = nn.get_io_pairs()

    data['lp_after'] = nn.to_lp()
    # data['lp_after_params'] = jsons.lp_params(lp1)

    return data


def run_loop():
    recipe_path = 'templates/example01.json'
    params = [
        {"ahln": 1, "bias": 0.0}
    ]

    for param in params:
        print(param)
        for i in range(1000):
            print(f"example {i}")
            d = loop(recipe_path, param)
            json_name = f"output_jsons2/{str(i).zfill(4)}_{str(param['ahln']).zfill(2)}_{param['bias']}.json"
            with open(json_name, 'w') as json_file:
                json.dump(d, json_file)


if __name__ == '__main__':
    run_loop()
