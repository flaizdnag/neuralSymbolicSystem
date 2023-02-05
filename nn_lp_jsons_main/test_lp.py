import unittest
from nn_lp_jsons_main.jsons import lp_params

lp_example = [
  {
    "tag": "Cl",
    "clHead": {
      "idx": 4,
      "label": ""
    },
    "clPAtoms": [
      {
        "idx": 2,
        "label": ""
      },
      {
        "idx": 1,
        "label": ""
      }
    ],
    "clNAtoms": [
      {
        "idx": 2,
        "label": ""
      }
    ]
  },
  {
    "tag": "Cl",
    "clHead": {
      "idx": 5,
      "label": "h"
    },
    "clPAtoms": [
    {
      "idx": 6,
      "label": "h"
    }
    ],
    "clNAtoms": []
  },
  {
    "tag": "Fact",
    "clHead": {
      "idx": 1,
      "label": ""
    },
    "clPAtoms": [],
    "clNAtoms": []
  },
  {
    "tag": "Cl",
    "clHead": {
      "idx": 1,
      "label": ""
    },
    "clPAtoms": [],
    "clNAtoms": [
      {
        "idx": 3,
        "label": ""
      },
      {
        "idx": 2,
        "label": ""
      }
    ]
  }
]

class Test(unittest.TestCase):
    def test_clauses(self):
        self.assertEqual(lp_params(lp_example)["clauses"]["amount"],3)
        self.assertEqual(lp_params(lp_example)["clauses"]["onlyPos"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["onlyNeg"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["mix"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["headWithH"],1)

    def test_atoms(self):
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["sum"],6)
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["pos"],3)
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["neg"],3)
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["withH"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["posWithH"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["atoms"]["negWithH"],0)

    def test_difAtoms(self):
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["sum"],4)
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["pos"],3)
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["neg"],2)
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["withH"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["posWithH"],1)
        self.assertEqual(lp_params(lp_example)["clauses"]["difAtoms"]["negWithH"],0)

    def test_facts(self):
        self.assertEqual(lp_params(lp_example)["facts"],1)

    def test_assumptions(self):
        self.assertEqual(lp_params(lp_example)["assumptions"],0)

if __name__ == '__main__':
    unittest.main()
