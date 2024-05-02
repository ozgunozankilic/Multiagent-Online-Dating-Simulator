import pytest
from catfish_sim import agents, compatibility, strategies
import copy
import math


def test_deceptive_attribute():
    """Ensures attribute misrepresentation affects the perceived compatibility scores."""

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
            value=180,
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
    male_hidden_attributes["Height"].value = 175

    male_agent = agents.Agent(
        id=0,
        reported_attributes=male_reported_attributes,
        hidden_attributes=male_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=5,
        estimated_attractiveness=5,
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
        estimated_attractiveness=1,
    )

    # Interested with misrepresentation.
    assert (
        female_agent.strategy.is_interested(
            agent=female_agent, candidate_details=male_agent.get_public_details()
        )
        == True
    )

    # Not interested without misrepresentation.
    male_agent.reported_attributes["Height"].value = 175
    assert (
        female_agent.strategy.is_interested(
            agent=female_agent, candidate_details=male_agent.get_public_details()
        )
        == False
    )


def test_deceptive_preference():
    """Ensures preference misrepresentation affects the perceived compatibility scores."""

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
        estimated_attractiveness=5,
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
                preferred_range=[175, 200],  # Not their real preference range
                allowed_range=[150, 210],
                preferred_score=1.25,
                nonpreferred_score=-math.inf,  # Very strict with height preference
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
    female_hidden_attributes["Height"].preference.preferred_range = [
        180,
        200,
    ]  # Their actual preference range

    female_agent = agents.Agent(
        id=1,
        reported_attributes=female_reported_attributes,
        hidden_attributes=female_hidden_attributes,
        like_allowance=100,
        strategy=strategies.WeightedMinimal(),
        compatibility_calculator=compatibility.CompatibilityCalculator(),
        attractiveness=1,
        estimated_attractiveness=1,
    )

    compatibility_calculator = compatibility.CompatibilityCalculator()

    # Considering only the reported attributes, the system believes the female agent would be interested.
    assert (
        compatibility_calculator.get_compatibility(
            female_agent.reported_attributes, male_agent.reported_attributes
        )
        >= 1
    )

    # The female agent is not actually interested beceause the male agent is too short.
    assert (
        female_agent.strategy.is_interested(
            agent=female_agent, candidate_details=male_agent.get_public_details()
        )
        == False
    )
