import unittest

import src
import src.neural_network.network as network
import numpy as np
import json

'''
EXAMPLE_RECIPE = {'inpLayer': [{'label': 'A1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'inp1'},
                               {'label': 'A2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'inp2'}],
                  'hidLayer': [{'label': 'ha1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'hid1'},
                               {'label': 'ha2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'hid2'}],
                  'outLayer': [{'label': 'A1', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'out1'},
                               {'label': 'A2', 'activFunc': 'idem', 'bias': 0.0, 'idx': 'out2'}],
                  'inpToHidConnections': [{'fromNeuron': 'inp1', 'toNeuron': 'hid1', 'weight': 0.1},
                                          {'fromNeuron': 'inp1', 'toNeuron': 'hid2', 'weight': 0.2},
                                          {'fromNeuron': 'inp2', 'toNeuron': 'hid1', 'weight': 0.3},
                                          {'fromNeuron': 'inp2', 'toNeuron': 'hid2', 'weight': 0.4}],
                  'hidToOutConnections': [{'fromNeuron': 'hid1', 'toNeuron': 'out1', 'weight': 0.1},
                                          {'fromNeuron': 'hid1', 'toNeuron': 'out2', 'weight': 0.2},
                                          {'fromNeuron': 'hid2', 'toNeuron': 'out1', 'weight': 0.3},
                                          {'fromNeuron': 'hid2', 'toNeuron': 'out2', 'weight': 0.4}],
                  'recConnections': [{'fromNeuron': 'out1', 'toNeuron': 'inp1', 'weight': 0.2},
                                     {'fromNeuron': 'out2', 'toNeuron': 'inp2', 'weight': 0.4}], 'nn_factors': []}
'''


def construct_network(lp_path: str = r'C:\Users\p.sowinski\Synchair\RRL\NeuralSymbolicSystem\templates\example01.json') -> network.NeuralNetwork3L:

    with open(lp_path, 'r') as json_file:
        recipe = json.load(json_file)

    lp = src.logic.LogicProgram.from_dict(recipe['lp'])
    ag = src.logic.Clause.from_dict(recipe['abductive_goal'])
    factors = src.logic.Factors.from_dict(recipe['factors'])

    nn_recipe = src.connect.get_nn_recipe(lp, ag, factors)
    return network.NeuralNetwork3L.from_dict(nn_recipe)


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth(self):
        self.assertEqual(1+1, 2)

    def test_all_false(self):

        amin = 0.433

        nn = construct_network()
        nn.factors.amin = amin

        output = nn.forward(nn.set_true([]))

        self.assertIsInstance(output, np.ndarray)
        self.assertTrue(all(map(lambda x: isinstance(x, float), output)))
        self.assertTrue(all(map(lambda x: x < amin, output)))

    def test_io(self, i: [str], o: [str]):
        amin = 0.433

        nn = construct_network()
        nn.factors.amin = amin

        output = {label: value for label, value in zip(nn.out_layer_spec.label, nn.forward(nn.set_true(i)))}

        self.assertTrue(all(map(lambda neuron: output[neuron] > amin, o)))

    def test_pairs(self):

        self.test_io(['A1'], ['A2'])
        self.test_io(['A3'], ['A2'])
        self.test_io(['A3', 'A2'], ['A1', 'A2'])
        self.test_io([], [])

if __name__ == '__main__':
    unittest.main()
