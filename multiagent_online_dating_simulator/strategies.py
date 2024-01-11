import random
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


class AdaptiveWeightedMinimal(Strategy):
    """An experimental version of the Weighted Minimal strategy. While the liking strategy is the same, this class
    changes the agent's reported preferences. Never used, needs testing, and this approach would probably be not
    fruitful.
    """

    name = "Adaptive Weighted Minimal"

    def __init__(self, evaluation_interval=5):
        """Initializes the strategy with the given homophily threshold.

        Args:
            evaluation_interval (int, optional): The number of rounds after which the agent evaluates and adapts their
                reported preferences. Defaults to 5.
        """
        self.optimization = optuna.create_study(direction="maximize")
        self.trial_backlog = dict()
        self.like_history = dict()
        self.like_rounds = dict()
        self.match_history = dict()
        self.happiness_history = dict()
        self.parameter_history = dict()
        self.evaluation_interval = evaluation_interval

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
            if agent.round not in self.like_history:
                self.like_history[agent.round] = []
            self.like_history[agent.round].append(candidate_details["id"])
            self.like_rounds[candidate_details["id"]] = agent.round
            return True
        else:
            return False

    def new_round_hook(self, agent):
        """A hook function used to notify the strategy about a new round. This is used for adaptive observed preference
        updates. The updates are done using Optuna, treating preferences like hyperparameters, but the approach was not
        very successful in preliminary experiments.

        Args:
            agent (Agent): The agent object that owns this strategy.
        """

        agent.round += 1
        self.trial_backlog[agent.round] = self.optimization.ask()
        self.happiness_history[agent.round] = 0
        self.parameter_history[agent.round] = dict()

        rounds_to_remove = []
        for round_id in self.trial_backlog.keys():
            if round_id + self.evaluation_interval < agent.round:
                self.optimization.tell(
                    self.trial_backlog[round_id], self.happiness_history[round_id]
                )
                rounds_to_remove.append(round_id)

        for round_id in rounds_to_remove:
            del self.trial_backlog[round_id]

        for attribute_name in agent.reported_attributes.keys():
            if agent.reported_attributes[
                attribute_name
            ].preference.allowed_values:  # Categorical
                pass  # Not touching categorical preferences for now
            else:  # Numerical
                allowed_min, allowed_max = agent.reported_attributes[
                    attribute_name
                ].preference.allowed_range

                allowed_deviation = 5

                if agent.hidden_attributes["Gender"].value == "Male":
                    new_max = self.trial_backlog[agent.round].suggest_int(
                        attribute_name + "_max",
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[1]
                        - allowed_deviation,
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[1]
                        + allowed_deviation,
                    )
                    new_min = self.trial_backlog[agent.round].suggest_int(
                        attribute_name + "_min",
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[0]
                        - allowed_deviation,
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[0]
                        + allowed_deviation,
                    )
                else:  # Female
                    new_min = self.trial_backlog[agent.round].suggest_int(
                        attribute_name + "_min",
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[0]
                        - allowed_deviation,
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[0]
                        + allowed_deviation,
                    )
                    new_max = self.trial_backlog[agent.round].suggest_int(
                        attribute_name + "_max",
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[1]
                        - allowed_deviation,
                        agent.reported_attributes[
                            attribute_name
                        ].preference.preferred_range[1]
                        + allowed_deviation,
                    )
                new_min = max(new_min, allowed_min)
                new_max = min(new_max, allowed_max)
                new_range = sorted([new_min, new_max])
                agent.reported_attributes[
                    attribute_name
                ].preference.preferred_range = new_range
                self.parameter_history[agent.round][attribute_name] = new_range

    def match_hook(self, agent, matched_agent):
        if agent.round not in self.match_history.keys():
            self.match_history[agent.round] = []
        self.match_history[agent.round].append(matched_agent.id)

        self.happiness_history[agent.round] += agent.calculate_happiness(
            matched_agent=matched_agent
        )


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
