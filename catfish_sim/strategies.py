import random
import numpy as np


class Strategy:
    """Strategy parent class."""

    def is_interested(self, agent, candidate_details):
        pass

    def match_hook(self, agent, matched_agent):
        pass

    def new_round_hook(self, agent):
        pass


class WeightedMinimal(Strategy):
    """A threshold-based strategy. Depending on the settings, if the candidate compatibility is 1 or higher, or the
    multiplication of candidate attractiveness and compatibility is equal or higher than the agent's estimated
    self-attractiveness, the agent likes the candidate. Currently, always considers attractiveness as well.
    """

    name = "Weighted Minimal"

    def is_interested(self, agent, candidate_details, consider_attractiveness=True):
        """Decides whether the given agent is interested in the given candidate. The agent also uses their hidden
        attributes for this, which supersedes the reported ones.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.
            consider_attractiveness (bool, optional): Indicates whether attractiveness will be considered as well.
                Defaults to True.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False) behavior.
        """
        combined_attributes = agent.reported_attributes.copy()
        combined_attributes.update(agent.hidden_attributes)
        attribute_compatibilities = []
        for key in combined_attributes.keys():
            if key in candidate_details["reported_attributes"]:
                attribute_compatibilities.append(
                    combined_attributes[key].preference.evaluate_attribute(
                        candidate_details["reported_attributes"][key].value
                    )
                )

        compatibility = np.mean(attribute_compatibilities)

        if (
            consider_attractiveness
            and compatibility * candidate_details["attractiveness"]
            >= agent.estimated_attractiveness
        ) or (not consider_attractiveness and compatibility >= 1):
            return True
        else:
            return False


class Adventurous(Strategy):
    """A strategy that randomly likes the candidate."""

    name = "Adventurous"

    def is_interested(self, agent, candidate_details):
        """_summary_

        Args:
            agent: Not used.
            candidate_details: Not used.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        return random.choice([True, False])


class PhysicalHomophiliac(Strategy):
    """A strategy that favors candidates who are in a specified attractiveness threshold
    with the agent's own estimated attractiveness.
    """

    name = "Physical Homophiliac"

    def __init__(self, homophily_threshold=[-1.5, 2]):
        """Initializes the strategy with the given homophily threshold.

        Args:
            homophily_threshold (list, optional): A [min, max] homophily threshold used
                to decide whether a candidate is similar enough. The difference between
                the agent's estimated attractiveness and the candidate's attractiveness
                must be within this inclusive threshold to be liked. Defaults to
                [-1.5, 2].
        """
        self.homophily_threshold = homophily_threshold

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        attractiveness_difference = (
            agent.estimated_attractiveness - candidate_details["attractiveness"]
        )
        if (
            self.homophily_threshold[0]
            <= attractiveness_difference
            <= self.homophily_threshold[1]
        ):
            return True
        else:
            return False


class SocialClimber(Strategy):
    """A strategy that strictly likes candidates who are more attractive than the
    agent's own estimated attractiveness."""

    name = "Social Climber"

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        if agent.estimated_attractiveness < candidate_details["attractiveness"]:
            return True
        else:
            return False
