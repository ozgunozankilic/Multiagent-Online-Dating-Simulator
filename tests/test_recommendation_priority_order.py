import pytest
from catfish_sim import (
    agents,
    compatibility,
    strategies,
    utils,
    matchers,
)
import copy
import math
import random
import numpy as np


def test_recommendation_priority_order():
    """Ensures that recommendation priority order strictly follows the compatibility order for PreferentialAgentMatcher.
    Repeats the test with different agent populations and judger weights.
    """

    n_agents = 10

    for judger_weight in np.arange(0, 1.01, 0.1):
        users = []
        for i in range(n_agents):
            gender = utils.get_random_gender()
            age = utils.sample_age_from_sex(gender)
            preferred_age_range = utils.sample_age_preference(gender, age)
            height = round(utils.sample_height_from_sex_age(gender, age))
            preferred_height_range = utils.get_height_preference(gender, height)
            bmi = utils.sample_bmi_from_sex_age(gender, age)
            preferred_bmi_range = utils.get_bmi_preference(gender, bmi)
            reported_attributes = {
                "Gender": compatibility.Attribute(
                    name="Gender",
                    value=gender,
                    preference=compatibility.CategoricalPreference(
                        preferred_values=[("Female" if gender == "Male" else "Male")],
                        allowed_values=["Male", "Female"],
                        preferred_score=1,
                        nonpreferred_score=-math.inf,
                    ),
                ),
                "Age": compatibility.Attribute(
                    name="Age",
                    value=age,
                    preference=compatibility.NumericalPreference(
                        preferred_range=preferred_age_range,
                        allowed_range=[1, 13],
                        preferred_score=1.25,
                        nonpreferred_score=0.25,
                        distance_sensitive=True,
                        compatibility_weight=1,
                    ),
                ),
                "Height": compatibility.Attribute(
                    name="Height",
                    value=height,
                    preference=compatibility.NumericalPreference(
                        preferred_range=preferred_height_range,
                        allowed_range=[150, 210],
                        preferred_score=1.25,
                        nonpreferred_score=0.25,
                        distance_sensitive=True,
                        compatibility_weight=1,
                    ),
                ),
                "Weight": compatibility.Attribute(
                    name="Weight",
                    value=bmi,
                    preference=compatibility.NumericalPreference(
                        preferred_range=preferred_bmi_range,
                        allowed_range=[1, 4],
                        preferred_score=1.25,
                        nonpreferred_score=0.25,
                        distance_sensitive=True,
                        compatibility_weight=1.5,
                    ),
                ),
            }
            new_agent = agents.Agent(
                id=i,
                reported_attributes=reported_attributes,
                hidden_attributes=copy.deepcopy(reported_attributes),
                like_allowance=100,
                strategy=strategies.WeightedMinimal(),
                compatibility_calculator=compatibility.CompatibilityCalculator(),
            )
            users.append(new_agent)

    preferential_matcher = matchers.PreferentialAgentMatcher(
        agents=users,
        recommendation_limit=100,
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        judger_weight=judger_weight,
        recalculate=False,
    )

    for agent in users:
        priority_queue = preferential_matcher.recommendation_priority[agent.id]
        last_compatibility = math.inf
        # If Candidate A comes after Candidate B, Candidate A cannot have a higher compatibility score.
        for candidate in priority_queue:
            match_compatibility = judger_weight * (
                preferential_matcher.compatibility_calculator.get_compatibility(
                    agent.reported_attributes,
                    preferential_matcher.agents[candidate].reported_attributes,
                )
            ) + (
                1 - judger_weight
            ) * preferential_matcher.compatibility_calculator.get_compatibility(
                preferential_matcher.agents[candidate].reported_attributes,
                agent.reported_attributes,
            )
            assert match_compatibility <= last_compatibility
            last_compatibility = match_compatibility
