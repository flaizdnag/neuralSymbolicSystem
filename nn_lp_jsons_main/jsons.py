import json


# summary of neural network parameters
def nn_params(source):
    if type(source) == str:
        with open(source, "r") as data:
            neural_network = json.load(data)
    else:
        neural_network = source

    nn_params = {"atoms":
                     {"sum": 0,
                      "inp": 0,
                      "hid": 0,
                      "out": 0,
                      "rec": 0},
                 "connections":
                     {"inp2Hid": 0,
                      "hid2Out": 0,
                      "rec": 0},
                 "bigWeights":
                     {"inp2Hid": 0,
                      "hid2Out": 0,
                      "rec": 0},
                 "factors": {}}

    # sums of atoms on each layer
    nn_params["atoms"]["sum"] = len((neural_network)["neuralNetwork"]["inpLayer"]) + len((neural_network)["neuralNetwork"]["hidLayer"]) + len(
        (neural_network)["neuralNetwork"]["outLayer"])
    nn_params["atoms"]["inp"] = len((neural_network)["neuralNetwork"]["inpLayer"])
    nn_params["atoms"]["hid"] = len((neural_network)["neuralNetwork"]["hidLayer"])
    nn_params["atoms"]["out"] = len((neural_network)["neuralNetwork"]["outLayer"])

    # sum of connections inp2Hid and hid2Out
    nn_params["connections"]["inp2Hid"] = len(neural_network["neuralNetwork"]["inpToHidConnections"])
    nn_params["connections"]["hid2Out"] = len(neural_network["neuralNetwork"]["hidToOutConnections"])

    # connections with 'bigWeights'
    for i in neural_network["neuralNetwork"]["inpToHidConnections"]:
        if abs(i["weight"]) >= neural_network["nnFactors"]["amin_factor"]:
            nn_params["bigWeights"]["inp2Hid"] += 1

    for i in neural_network["neuralNetwork"]["hidToOutConnections"]:
        if abs(i["weight"]) >= neural_network["nnFactors"]["amin_factor"]:
            nn_params["bigWeights"]["hid2Out"] += 1

    # atoms and connections on recLayer
    if "recLayer" in (neural_network)["neuralNetwork"]:
        nn_params["atoms"]["sum"] += len((neural_network)["neuralNetwork"]["recLayer"])
        nn_params["atoms"]["rec"] = len((neural_network)["neuralNetwork"]["recLayer"])  # atoms in recLayer
        nn_params["connections"]["rec"] = len(neural_network["neuralNetwork"]["recConnections"])  # recConnections
        for i in neural_network["neuralNetwork"]["recConnections"]:
            if abs(i["weight"]) >= neural_network["nnFactors"]["amin_factor"]:
                nn_params["bigWeights"]["rec"] += 1  # 'bigWeights' of recConnections

    # neural network factors
    nn_params["factors"] = neural_network["nnFactors"]

    return nn_params


# summary od logic program parameters
def lp_params(source):
    if type(source) == str:
        with open(source, "r") as data:
            logic_program = json.load(data)
    else:
        logic_program = source

    lp_params = {"clauses": {
        "amount": 0,
        "onlyPos": 0,
        "onlyNeg": 0,
        "mix": 0,
        "headWithH": 0,
        "atoms": {"sum": 0, "pos": 0, "neg": 0, "withH": 0, "posWithH": 0, "negWithH": 0},
        "difAtoms": {"sum": 0, "pos": 0, "neg": 0, "withH": 0, "posWithH": 0, "negWithH": 0},
        "atomsInHeadsNotBodies": [],
        "atomsInBodiesNotInHeads": [],
        "numOfPosAtomsInEachClause": [],
        "numOfNegAtomsInEachClause": [],
        "numOfClausesWhoseHeadAppearsInTheBody": 0,
        "numOfClausesWhoseHeadAppearsInABody": 0
    },
        "facts": 0,
        "assumptions": 0}

    # lists to count parameters
    atoms = []
    atoms_pos = []
    atoms_neg = []
    atoms_heads = []
    atoms_bodies = []

    # CLAUSES PARAMS
    #if isinstance(logic_program, list):
    #    logic_program = { "lp" : logic_program }
    for i in range(len(logic_program["lp"]["clauses"])):
        # main params
        if logic_program["lp"]["clauses"][i]["tag"] == "Cl":
            lp_params["clauses"]["amount"] += 1  # sum of clauses
            atoms_heads.append(logic_program["lp"]["clauses"][i]["clHead"])

            if len(logic_program["lp"]["clauses"][i]["clPAtoms"]) == 0:  # clauses with only positive atoms
                lp_params["clauses"]["onlyNeg"] += 1
            elif len(logic_program["lp"]["clauses"][i]["clNAtoms"]) == 0:  # clauses with only negative atoms
                lp_params["clauses"]["onlyPos"] += 1
            else:  # mix clauses
                lp_params["clauses"]["mix"] += 1
                lp_params["clauses"]["atoms"]["sum"] += len(logic_program["lp"]["clauses"][i]["clPAtoms"]) + len(
                    logic_program["lp"]["clauses"][i]["clNAtoms"])  # suma klauzul

            # clauses with h in the head
            if logic_program["lp"]["clauses"][i]["clHead"]["label"] == "h":
                lp_params["clauses"]["headWithH"] += 1

            # ATOMS PARAMS
            clause_body = []
            # positive atoms
            for j in logic_program["lp"]["clauses"][i]["clPAtoms"]:  # sum
                lp_params["clauses"]["atoms"]["pos"] += 1
                atoms_bodies.append(j)
                clause_body.append(j)

                if j["label"] == "h":  # positive atom with h
                    lp_params["clauses"]["atoms"]["withH"] += 1
                    lp_params["clauses"]["atoms"]["posWithH"] += 1
                if j not in atoms_pos:
                    atoms_pos.append(j)
                    lp_params["clauses"]["difAtoms"]["pos"] += 1  # positive atoms without repeats
                if not j in atoms:
                    atoms.append(j)
                    if j["label"] == "h":  # positive atoms with h without repeats
                        lp_params["clauses"]["difAtoms"]["withH"] += 1
                        lp_params["clauses"]["difAtoms"]["posWithH"] += 1

            # negative atoms
            for j in logic_program["lp"]["clauses"][i]["clNAtoms"]:  # sum
                lp_params["clauses"]["atoms"]["neg"] += 1
                atoms_bodies.append(j)
                clause_body.append(j)

                if j["label"] == "h":  # negative atoms with h
                    lp_params["clauses"]["atoms"]["withH"] += 1
                    lp_params["clauses"]["atoms"]["negWithH"] += 1
                if j not in atoms_neg:
                    atoms_neg.append(j)
                    lp_params["clauses"]["difAtoms"]["neg"] += 1  # negative atoms without repeats
                if not j in atoms:
                    atoms.append(j)
                    if j["label"] == "h":  # negative atoms with h without repeats
                        lp_params["clauses"]["difAtoms"]["withH"] += 1
                        lp_params["clauses"]["difAtoms"]["negWithH"] += 1

            # all atoms without repeats
            lp_params["clauses"]["difAtoms"]["sum"] = len(atoms)

        # number of clauses whose head appears in the body
        if logic_program["lp"]["clauses"][i]["clHead"] in clause_body:
            lp_params["clauses"]["numOfClausesWhoseHeadAppearsInTheBody"] += 1

        # number od positive/negative atoms in each clause
        lp_params["clauses"]["numOfPosAtomsInEachClause"].append(len(logic_program["lp"]["clauses"][i]["clPAtoms"]))
        lp_params["clauses"]["numOfNegAtomsInEachClause"].append(len(logic_program["lp"]["clauses"][i]["clNAtoms"]))

    # atoms in heads not in bodies
    for i in atoms_heads:
        if not i in atoms_bodies:
            lp_params["clauses"]["atomsInHeadsNotBodies"].append(i)

    # atoms in bodies not in heads
    for i in atoms_bodies:
        if not i in atoms_heads:
            lp_params["clauses"]["atomsInBodiesNotInHeads"].append(i)

    # number of Clauses whose head appears in a body
    for i in logic_program["lp"]["clauses"]:
        if i["clHead"] in atoms_bodies:
            lp_params["clauses"]["numOfClausesWhoseHeadAppearsInABody"] += 1

    # FACTS AND ASSUMPTIONS
    lp_params["facts"] += len(logic_program["lp"]["facts"])  # fakty
    lp_params["assumptions"] += len(logic_program["lp"]["assumptions"])  # assumptions

    return lp_params
