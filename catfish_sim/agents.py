import scipy.stats as stats
import random
from sklearn.preprocessing import minmax_scale


class Agent:
    """The base agent class that can take different attributes."""

    def __init__(
        self,
        id,
        reported_attributes,
        hidden_attributes,
        like_allowance,
        strategy,
        compatibility_calculator,
        attractiveness=None,
        estimated_attractiveness=None,
    ):
        """Initializes the agent with the provided attributes.

        Args:
            id (int): Agent ID.
            reported_attributes (list): A list of attributes reported attributes that are shown to the system and other
                agents.
            hidden_attributes (list): A list of attributes that is added to the reported ones and possibly overrides
                them when the agent is making decisions. Used when an agent is not truthful or some attributes are never
                reported.
            like_allowance (int): The number agents that the agent can like in a day (round).
            strategy (Strategy): A Strategy object that will be used to like or pass the candidates.
            compatibility_calculator (CompatibilityCalculator): A CompatibilityCalculator object to find the
                compatibility with an agent.
            attractiveness (float, optional): An attractiveness level between 1 and 5. If None, sampled based on the
                gender. Defaults to None.
            estimated_attractiveness (float, optional): An estimated attractiveness level between 1 and 5. If None,
                sampled based on the actual attractiveness. Defaults to None.
        """
        self.id = id
        self.reported_attributes = reported_attributes
        self.hidden_attributes = hidden_attributes
        self.like_allowance = like_allowance
        self.remaining_likes = like_allowance
        self.strategy = strategy
        self.compatibility_calculator = compatibility_calculator
        self.attractiveness = (
            attractiveness if attractiveness else self.get_random_attractiveness()
        )
        self.estimated_attractiveness = (
            estimated_attractiveness
            if estimated_attractiveness
            else self.estimate_self_attractiveness()
        )
        self.liked = set()
        self.passed = set()
        self.matched = set()
        self.match_count = 0
        self.happiness = 0
        self.history = {}
        self.round = 0

    def get_random_attractiveness(self):
        """Randomly samples the attractiveness level from gender-specific beta distributions.

        Returns:
            float: Attractiveness.
        """
        if self.hidden_attributes["Gender"].value == "Male":
            a, b = 2, 6
        else:  # Female
            a, b = 4, 4
        return random.choice(
            minmax_scale(stats.beta.rvs(a, b, size=1000), feature_range=(1, 5))
        )

    def estimate_self_attractiveness(self):
        """Estimates its own attractiveness with an error margin. Agents cannot use their actual attractiveness levels
        for anything. Instead, they use their estimations (unless they use a strategy that updates this estimation).

        Returns:
            float: Estimated self-attractiveness.
        """
        estimation_max = (3 * self.attractiveness / 8) + 25 / 8
        estimation_min = (3 * self.attractiveness / 4) + 1 / 4
        mu = (estimation_max + estimation_min) / 2
        sigma = (estimation_max - estimation_min) / 6
        distribution = stats.truncnorm(
            (estimation_min - mu) / sigma,
            (estimation_max - mu) / sigma,
            loc=mu,
            scale=sigma,
        )
        return distribution.rvs()

    def get_public_details(self):
        """Returns the agent's ID, attractivenes, and reported attributes which are passed to the candidates that
        evaluate the agent.

        Returns:
            dict: A dictionary that includes the ID, attractiveness, and reported attributes.
        """
        return {
            "id": self.id,
            "attractiveness": self.attractiveness,
            "reported_attributes": self.reported_attributes,
        }

    def get_assessed_candidates(self):
        """Returns the previously encountered (liked or passed) candidates' IDs.

        Returns:
            set: A set of agent IDs.
        """
        return self.liked.union(self.passed)

    def assess_candidates(self, candidates_details):
        """Classifies the provided candidates into liked and passed candidate groups.

        Args:
            candidates_details (list): A list of dictionaries for public candidates' details.

        Returns:
            dict: A dictionary of "liked" and "passed" candidates.
        """
        liked = set()
        passed = set()
        for candidate_details in candidates_details:
            if self.remaining_likes < 1:
                break

            interested = self.strategy.is_interested(
                agent=self, candidate_details=candidate_details
            )
            if interested:
                self.remaining_likes -= 1
                liked.add(candidate_details["id"])
            else:
                passed.add(candidate_details["id"])

        self.liked.update(liked)
        self.passed.update(passed)
        return {"liked": liked, "passed": passed}

    def calculate_happiness(self, matched_agent):
        """Returns the match happiness (utility) for a given matched agent using the utility formula. The utility
        suffers from diminishing returns.

        Args:
            matched_agent (Agent): The matched agent's object.

        Returns:
            float: Match happiness.
        """
        combined_attributes = self.reported_attributes.copy()
        combined_attributes.update(self.hidden_attributes)
        match_attributes = matched_agent.reported_attributes.copy()
        match_attributes.update(matched_agent.hidden_attributes)
        compatibility = 0
        evaluated_attributes = 0
        for key in combined_attributes.keys():
            if key in match_attributes.keys():
                compatibility += combined_attributes[key].preference.evaluate_attribute(
                    matched_agent.reported_attributes[key].value
                )
                evaluated_attributes += 1

        compatibility /= evaluated_attributes

        happiness = (matched_agent.attractiveness * compatibility + 2) ** (0.9) - 1

        if (
            self.hidden_attributes["Gender"].value == "Female"
            and matched_agent.attractiveness - self.estimated_attractiveness > 0
        ):
            discomfort_divisor = (
                (5 - self.estimated_attractiveness)
                / (5 * self.estimated_attractiveness)
            ) + (matched_agent.attractiveness - self.estimated_attractiveness) ** (
                1 / 100
            )
            happiness /= discomfort_divisor

        return happiness

    def get_matched(self, matched_agent):
        """Informs the agent about the match with the provided agent.

        Args:
            matched_agent (Agent): The matched agent's object.

        Returns:
            bool: Success and acknowledgment indicator.
        """
        self.match_count += 1
        self.matched.add(matched_agent.id)
        self.happiness += self.calculate_happiness(matched_agent=matched_agent) * (
            0.999**self.match_count
        )
        self.strategy.match_hook(self, matched_agent)
        return True

    def log_state(self, log_id, logged_variables):
        """Logs the agent's current state with the given log (day/round) ID and the requested log variables.

        Args:
            log_id (int): Round ID that will also uniquely identify the log.
            logged_variables (list): A list of variable names from the agent object.

        Returns:
            bool: Success and acknowledgment indicator.
        """
        for variable in logged_variables:
            if hasattr(self, variable):
                if variable not in self.history:
                    self.history[variable] = {log_id: self.__dict__[variable]}
                else:
                    self.history[variable][log_id] = self.__dict__[variable]
        return True

    def get_logs(self, log_id=None, variables=None):
        """Retrieves the agent logs.

        Args:
            log_id (int, optional): The round state that will be returned. If None, all round logs are returned.
                Defaults to None.
            variables (list, optional): The set of variables that will be returned. If None, all available variables are
                returned. Defaults to None.

        Returns:
            dict: A dictionary of variables that have the round values.
        """
        if log_id is None and variables is None:
            return self.history
        elif log_id is not None and variables is None:
            return {
                variable: self.history[variable][log_id]
                for variable in self.history.keys()
                if log_id in self.history[variable].keys()
            }
        elif log_id is None and variables is not None:
            return {
                variable: self.history[variable]
                for variable in variables
                if variable in self.history.keys()
            }
        elif log_id is not None and variables is not None:
            return {
                variable: self.history[variable][log_id]
                for variable in variables
                if variable in self.history.keys()
                and log_id in self.history[variable].keys()
            }
