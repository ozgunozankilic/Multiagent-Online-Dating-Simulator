# Multiagent Online Dating Simulator

This is a reasonably extensible simulation framework used in the paper "Catfished! Impacts of Strategic Misrepresentation in Online Dating" written by Oz Kilic and Alan Tsang.

The framework creates a Tinder-like virtual online dating environment where the primary focus is to simulate the swiping and mate choice behavior of people. Agents can have different genders, attractiveness levels, reported/hidden attributes, reported/hidden preferences, liking strategies, and compatibility assesment methods. These differences significantly impact the agents' utility (happiness) which is derived from both the quantity and quality of matches (matching with more attractive and more compatible agents yields higher happiness in general). It features various matchmaking systems, including those based on ratings, compatibility, or random selection, presenting agents with candidates to like or pass.

By default, some values, functions, and distributions are modeled after existing studies on dating and romantic relationships. Due to the constraints of existing studies and for the sake of simplicity, the simulator currently models cis-heteronormative interactions. We recognize these limitations. The framework is designed to be flexible, allowing for the incorporation of additional genders and sexual orientations with minimal modifications.

If you use this framework or our paper, please cite our paper:

```
@inproceedings{kilic2024catfished,
  title={Catfished! Impacts of Strategic Misrepresentation in Online Dating},
  author={Kilic, Oz and Tsang, Alan},
  booktitle={Proceedings of the 2024 International Conference on Autonomous Agents and Multiagent Systems},
  year={2024}
}
```