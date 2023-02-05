import requests
import json
from urllib.request import urlopen, Request

import src.logic

# http://207.154.220.61:10099/api/getNN


def post(url: str, d: dict) -> dict:
    return json.loads(requests.post(url, json=d).text)


def post_json_file(url: str, path: str) -> dict:
    # As an example
    with open(path, 'r') as json_file:
        return post(url, d=json.load(json_file))


def post_json_file_and_save_to_file(url: str, path_from: str, path_to: str):
    # As an example

    received_json = post_json_file(url, path_from)

    with open(path_to, 'w') as to_json_file:
        json.dump(received_json, to_json_file)


# def get(f, phrase, url='http://207.154.220.61:10099/api/'):
# def get(f, phrase, url='http://64.225.103.216:10100/api/'):
def get(f, phrase, url='http://127.0.0.1:10100/api/'):
    """
    Opens url using Request library

    :param f: to which function you want to connect (Str)
    :param phrase: request phrase (Str)
    :param url: url of server (Str)
    :return: response (Str)

    """
    request = Request(url+f, phrase.encode("utf-8"))
    # print(request.get_full_url())
    response = urlopen(request)
    html = response.read()
    response.close()
    return html.decode("utf-8")


def get_nn_recipe(logic_program: src.logic.LogicProgram,
                  abductive_goal: src.logic.Clause,
                  factors: src.logic.Factors) -> dict:
    """
    Get a Neural Network Recipe from API.

    :param logic_program: logic program (src.logic.LogicProgram)
    :param abductive_goal: abductive goal (src.logic.Clause)
    :param factors: factors for neural network (src.logic.Factors)
    :return: recipe for neural network (dict)

    """
    request_dict = {"lp": logic_program.to_dict(),
                    "abductive_goal": abductive_goal.to_dict(),
                    "factors": factors.to_dict()}

    request_json = json.dumps(request_dict)
    return json.loads(get('lp2nn', request_json))


def get_lp_from_nn(
        order_inp: [str],
        order_out: [str],
        amin: float,
        io_pairs: [tuple]) -> dict:
    """
    Sends request for trasnlation of a neural network to a logic program.

    :param order_inp: list of atoms as they occur in the input layer
    :param order_out: list of atoms as they occur in the output layer
    :param amin: amin factor for the neural network
    :param io_pairs: list of pairs (input, output) for brutforce translation
    """

    request_dict = {"orderInp": list(map(str_to_atom, order_inp)),
                    "orderOut": list(map(str_to_atom, order_out)),
                    "amin": amin,
                    "ioPairs": io_pairs}

    request_json = json.dumps(request_dict)  # .replace('"', r'\"')
    response = get('nn2lp', request_json)
    return json.loads(response)


def str_to_atom(str_atom: str) -> str:
    """
    Creates a string that can be easily understood by haskell from a short
    string describing an atom.

    :param str_atom: short atom string, e.g. 'A1', 'A1^n'
    """
    if str_atom == "inpT":
        return str_atom

    atom_values = str_atom.split("A")[1]

    if "^" in atom_values:
        idx, label = atom_values.split("^")
    else:
        idx = atom_values
        label = ""

    return 'A {idx = ' + idx + ', label = ' + str(list(label)) + '}'
