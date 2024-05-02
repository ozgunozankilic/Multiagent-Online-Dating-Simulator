# Catfish-Sim: Multiagent Online Dating Simulator

This is a reasonably extensible multiagent simulation framework used in the paper _"Catfished! Impacts of Strategic Misrepresentation in Online Dating"_ written by Oz Kilic and Alan Tsang. It can be used to create a Tinder-like virtual online dating environment where the primary focus is to simulate the swiping and mate choice behavior of people with different attributes, preferences, strategies, and outcomes.

## Features

Agents have attractiveness (average perceived attractiveness from others' perspective), estimated attractiveness (their own perception of their attractiveness), attributes, preferences, compatibility assesment methods, and candidate evaluation strategies. They derive utility (happiness) from both the quantity and quality of matches (matching with more attractive and more compatible agents yields higher happiness in general). 

Agent attributes can be numeric or categorical. Agent preferences can have a continuous range (numeric) or discrete preferred values (categorical), along with attribute-specific importance and different options, such as how compatibility is calculated if a candidate's value falls outside the preferred range. An agent's reported attributes and preferences may not be truthful, which can manipulate other agents' and the matchmaking logic. However, after two agents mutually like each other and are informed of this match, the truthful attributes are exposed, which, along with the truthful preferences, determines the ultimate utility.

The framework has various matchmaking systems (also referred to as "matchers"), including those based on ratings, compatibility, or random selection, presenting agents with candidates to like or pass. For each round, each agent has a recommendation queue of a predefined size, from which they can like only a predefined number of candidates. Different matchers offer specific options. For example, the compatibility-based matcher can be configured to take the weighted average from both parties' perspective based on their reported preferences.

By default, some values, functions, and distributions are modeled after existing studies on dating and romantic relationships. Due to the constraints of existing studies and for the sake of simplicity, the simulator currently models cis-heteronormative interactions. We recognize these limitations. New agent/matchmaking logic can be implemented by passing custom functions, code modifications, or writing custom classes.

## Usage

### Installation

```bash
pip install catfish-sim
```

Dependencies:
*   Python >= 3.6
*   numpy >= 1.23.3
*   pandas >= 1.5.0
*   scikit-learn >= 1.3.2
*   scipy >= 1.8.1
*   trueskill >= 0.4.5

Older versions of some of the libraries are likely to work, but have not been tested. The Python version used in development was 3.10, although the software does not currently use any features implemented after version 3.6.

### Examples

Example code will be added soon.

## License

This package is licensed under the [GNU General Public License v3 (GPL-3)](https://github.com/ozgunozankilic/catfish-sim/blob/main/LICENSE).

## Citation

If you use this framework or our paper, please cite our paper:

```bibtex
@inproceedings{kilic2024catfished,
	author = {Kilic, Oz and Tsang, Alan},
	title = {Catfished! Impacts of Strategic Misrepresentation in Online Dating},
	year = {2024},
	publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
	booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
	pages = {1011–1019},
	isbn = {9798400704864},
}
```