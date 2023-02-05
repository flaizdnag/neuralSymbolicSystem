import unittest
from jsons import nn_params

nn_example = {
  "nn": {
    "inpLayer": [
      {
        "label": "A3",
        "activFunc": "idem",
        "bias": 0,
        "idx": "inp1"
      },
      {
        "label": "A1",
        "activFunc": "idem",
        "bias": 0,
        "idx": "inp2"
      },
      {
        "label": "A2",
        "activFunc": "idem",
        "bias": 0,
        "idx": "inp3"
      }
    ],
    "hidLayer": [
      {
        "label": "h1",
        "activFunc": "tanh",
        "bias": 11.250353,
        "idx": "hid1"
      },
      {
        "label": "h2",
        "activFunc": "tanh",
        "bias": 0,
        "idx": "hid2"
      },
      {
        "label": "hidT",
        "activFunc": "tanh",
        "bias": 0,
        "idx": "hidT"
      },
      {
        "label": "ha1",
        "activFunc": "tanh",
        "bias": 0,
        "idx": "hid3"
      }
    ],
    "outLayer": [
      {
        "label": "A3",
        "activFunc": "tanh",
        "bias": 0,
        "idx": "out1"
      },
      {
        "label": "A1",
        "activFunc": "tanh",
        "bias": 0,
        "idx": "out2"
      }
    ],
    "recLayer": [
      {
        "label": "recA2",
        "activFunc": "k",
        "bias": 0,
        "idx": "recA2"
      }
    ],
    "inpToHidConnections": [
      {
        "fromNeuron": "inp3",
        "toNeuron": "hid1",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "inp1",
        "toNeuron": "hid1",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "inp4",
        "toNeuron": "hid1",
        "weight": -7.0314703
      },
      {
        "fromNeuron": "inp1",
        "toNeuron": "hid2",
        "weight": 0.00011
      },
      {
        "fromNeuron": "inpT",
        "toNeuron": "hidT",
        "weight": 7.0314703
      }
    ],
    "hidToOutConnections": [
      {
        "fromNeuron": "hidT",
        "toNeuron": "out1",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "hid1",
        "toNeuron": "out2",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "hid2",
        "toNeuron": "out3",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "hidT",
        "toNeuron": "out5",
        "weight": 7.0314703
      },
      {
        "fromNeuron": "hid3",
        "toNeuron": "out2",
        "weight": -0.009050451
      },
      {
        "fromNeuron": "hid4",
        "toNeuron": "out3",
        "weight": -0.004610043
      },
      {
        "fromNeuron": "hidT",
        "toNeuron": "out3",
        "weight": -0.01267628
      }
    ],
    "recConnections": [
      {
        "fromNeuron": "out1",
        "toNeuron": "inp1",
        "weight": 1
      },
      {
        "fromNeuron": "out2",
        "toNeuron": "inp2",
        "weight": 0.0012
      },
      {
        "fromNeuron": "out4",
        "toNeuron": "inp4",
        "weight": 1
      }
    ]
  },
  "nnFactors": {
    "beta": 1,
    "ahln": 1,
    "r": 0.05,
    "bias": 0,
    "w": 0.1,
    "amin": 0.1
  }
}


class Test(unittest.TestCase):
    def test_atoms(self):
        self.assertEqual(nn_params(nn_example)["atoms"]["sum"],10)
        self.assertEqual(nn_params(nn_example)["atoms"]["inp"],3)
        self.assertEqual(nn_params(nn_example)["atoms"]["hid"],4)
        self.assertEqual(nn_params(nn_example)["atoms"]["out"],2)
        self.assertEqual(nn_params(nn_example)["atoms"]["rec"],1)

    def test_connections(self):
        self.assertEqual(nn_params(nn_example)["connections"]["inp2Hid"],5)
        self.assertEqual(nn_params(nn_example)["connections"]["hid2Out"],7)
        self.assertEqual(nn_params(nn_example)["connections"]["rec"],3)

    def test_bigWeights(self):
        self.assertEqual(nn_params(nn_example)["bigWeights"]["inp2Hid"],4)
        self.assertEqual(nn_params(nn_example)["bigWeights"]["hid2Out"],4)
        self.assertEqual(nn_params(nn_example)["bigWeights"]["rec"],2)

if __name__ == '__main__':
    unittest.main()
