import pytest
from catfish_sim import (
    agents,
    compatibility,
    strategies,
    matchers,
)
import copy
import math


def test_matchmaker():
    """Ensures that the matchmaker does not recommend an agent to another (and vice versa if the judger's weight is
    smaller than 1) if the judged agent has a deal-breaker attribute that immediately causes a pass to prevent
    preference-wise impossible candidates (when the compatibility is negative).
    """

    male_reported_attributes = {
        "Gender": compatibility.Attribute(
            name="Gender",
            value="Male",
            preference=compatibility.CategoricalPreference(
                preferred_values="Female",
                allowed_values=["Male", "Female"],
                preferred_score=1,
                nonpreferred_score=-math.inf,
            ),
        ),
        "Age": compatibility.Attribute(
            name="Age",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 3],
                allowed_range=[1, 13],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Height": compatibility.Attribute(
            name="Height",
            value=175,
            preference=compatibility.NumericalPreference(
                preferred_range=[150, 175],
                allowed_range=[150, 210],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Weight": compatibility.Attribute(
            name="Weight",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 2],
                allowed_range=[1, 4],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1.5,
            ),
        ),
    }

    male_hidden_attributes = copy.deepcopy(male_reported_attributes)

    male_agent = agents.Agent(
        id=0,
        reported_attributes=male_reported_attributes,
        hidden_attributes=male_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=5,
    )

    female_reported_attributes = {
        "Gender": compatibility.Attribute(
            name="Gender",
            value="Female",
            preference=compatibility.CategoricalPreference(
                preferred_values="Male",
                allowed_values=["Male", "Female"],
                preferred_score=1,
                nonpreferred_score=-math.inf,
            ),
        ),
        "Age": compatibility.Attribute(
            name="Age",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 3],
                allowed_range=[1, 13],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Height": compatibility.Attribute(
            name="Height",
            value=175,
            preference=compatibility.NumericalPreference(
                preferred_range=[180, 200],
                allowed_range=[150, 210],
                preferred_score=1.25,
                nonpreferred_score=-math.inf,  # Strictly prefers taller men
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Weight": compatibility.Attribute(
            name="Weight",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 2],
                allowed_range=[1, 4],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1.5,
            ),
        ),
    }

    female_hidden_attributes = copy.deepcopy(female_reported_attributes)

    female_agent = agents.Agent(
        id=1,
        reported_attributes=female_reported_attributes,
        hidden_attributes=female_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=1,
    )

    selfish_preferential_matcher = matchers.PreferentialAgentMatcher(
        agents=[male_agent, female_agent],
        recommendation_limit=100,
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        judger_weight=1,
        recalculate=False,
    )

    # The matcher only cares about the judger's viewpoint, so the male has a candidate while the female does not.
    assert len(selfish_preferential_matcher.recommendation_priority[0]) == 1
    assert len(selfish_preferential_matcher.recommendation_priority[1]) == 0

    twosided_preferential_matcher = matchers.PreferentialAgentMatcher(
        agents=[male_agent, female_agent],
        recommendation_limit=100,
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        judger_weight=0.99,
        recalculate=False,
    )

    # The matcher cares about both viewpoints, so no agent has a candidate, as one side of the match is incompatible.
    assert len(twosided_preferential_matcher.recommendation_priority[0]) == 0
    assert len(twosided_preferential_matcher.recommendation_priority[1]) == 0


def test_preference():
    """Ensures that a non-preferred value with strict misrepresentation affects the perceived compatibility scores."""

    male_reported_attributes = {
        "Gender": compatibility.Attribute(
            name="Gender",
            value="Male",
            preference=compatibility.CategoricalPreference(
                preferred_values="Female",
                allowed_values=["Male", "Female"],
                preferred_score=1,
                nonpreferred_score=-math.inf,
            ),
        ),
        "Age": compatibility.Attribute(
            name="Age",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 3],
                allowed_range=[1, 13],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Height": compatibility.Attribute(
            name="Height",
            value=175,
            preference=compatibility.NumericalPreference(
                preferred_range=[150, 175],
                allowed_range=[150, 210],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Weight": compatibility.Attribute(
            name="Weight",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 2],
                allowed_range=[1, 4],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1.5,
            ),
        ),
    }

    male_hidden_attributes = copy.deepcopy(male_reported_attributes)

    male_agent = agents.Agent(
        id=0,
        reported_attributes=male_reported_attributes,
        hidden_attributes=male_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=5,
    )

    female_reported_attributes = {
        "Gender": compatibility.Attribute(
            name="Gender",
            value="Female",
            preference=compatibility.CategoricalPreference(
                preferred_values="Male",
                allowed_values=["Male", "Female"],
                preferred_score=1,
                nonpreferred_score=-math.inf,
            ),
        ),
        "Age": compatibility.Attribute(
            name="Age",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 3],
                allowed_range=[1, 13],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Height": compatibility.Attribute(
            name="Height",
            value=175,
            preference=compatibility.NumericalPreference(
                preferred_range=[180, 200],
                allowed_range=[150, 210],
                preferred_score=1.25,
                nonpreferred_score=-math.inf,  # Strictly prefers taller men
                distance_sensitive=True,
                compatibility_weight=1,
            ),
        ),
        "Weight": compatibility.Attribute(
            name="Weight",
            value=2,
            preference=compatibility.NumericalPreference(
                preferred_range=[1, 2],
                allowed_range=[1, 4],
                preferred_score=1.25,
                nonpreferred_score=0.25,
                distance_sensitive=True,
                compatibility_weight=1.5,
            ),
        ),
    }

    female_hidden_attributes = copy.deepcopy(female_reported_attributes)

    female_agent = agents.Agent(
        id=1,
        reported_attributes=female_reported_attributes,
        hidden_attributes=female_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=1,
    )

    # Not interested.
    assert (
        female_agent.strategy.is_interested(
            agent=female_agent, candidate_details=male_agent.get_public_details()
        )
        == False
    )

    compatibility_calculator = compatibility.CompatibilityCalculator()

    # Compatibility would be negative.
    assert (
        compatibility_calculator.get_compatibility(
            female_agent.reported_attributes, male_agent.reported_attributes
        )
        < 0
    )
