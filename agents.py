import scipy.stats as stats
import random
from sklearn.preprocessing import minmax_scale


class Agent:
    """The base agent class that can take different attributes."""

    def __init__(
        self,
        id,
        observable_attributes,
        hidden_attributes,
        like_allowance,
        strategy,
        compatibility_calculator,
        gender=None,
        attractiveness=None,
        is_premium=0.05,
        is_impostor=0.1,
    ):
        """Initializes the agent with the provided attributes.

        Args:
            id (int): Agent ID.
            observable_attributes (list): A list of categorical attributes.
            hidden_attributes (list): A list of categorical attributes.
            like_allowance (int): The number agents that the agent can like in a round.
            strategy (Strategy): A Strategy object that will be used to like or pass the
                candidates.
            compatibility_calculator (CompatibilityCalculator): A
                CompatibilityCalculator object to find the compatibility with an agent.
            gender (str, optional): "Male" or "Female". If None, randomly assigned.
                Defaults to None.
            attractiveness (float, optional): An attractiveness level between 1 and 5.
                If None, sampled based on the gender. Defaults to None.
            is_premium (float, optional): The probability of the agent to be a premium
                member. Premium members' like allowance is doubled at each round.
                Defaults to 0.05. Use 0 for deterministically non-premium agents and 1
                for deterministically premium agents.
            is_impostor (float, optional): The probability of the agent being an
                impostor and not yielding any happiness to other agents unless they are
                impostors too. Defaults to 0.1. Use 0 for deterministically non-impostor
                agents and 1 for deterministically impostor agents.
        """
        self.id = id
        self.gender = gender if gender else self.get_random_gender()
        self.attractiveness = (
            attractiveness if attractiveness else self.get_random_attractiveness()
        )
        self.estimated_attractiveness = self.estimate_self_attractiveness()
        self.is_premium = (
            is_premium
            if type(is_premium) == bool
            else self.get_random_premiumness(is_premium)
        )
        self.is_impostor = (
            is_impostor
            if type(is_impostor) == bool
            else self.get_random_impostorness(is_impostor)
        )
        self.observable_attributes = observable_attributes
        self.hidden_attributes = hidden_attributes
        self.like_allowance = like_allowance * 2 if self.is_premium else like_allowance
        self.remaining_likes = like_allowance
        self.strategy = strategy
        self.compatibility_calculator = compatibility_calculator
        self.liked = set()
        self.passed = set()
        self.matched = set()
        self.match_count = 0
        self.happiness = 0
        self.history = {}

    def get_random_gender(self):
        """Randomly assigns the gender."""
        return random.choices(population=["Male", "Female"], weights=[0.72, 0.28], k=1)[
            0
        ]

    def get_random_attractiveness(self):
        """Randomly samples the attractiveness level from gender-specific attractiveness
        distributions.
        """
        if self.gender == "Male":
            a, b = 2, 6
        else:  # Female
            a, b = 4, 4
        return random.choice(
            minmax_scale(stats.beta.rvs(a, b, size=1000), feature_range=(1, 5))
        )

    def get_random_premiumness(self, chance):
        """Randomly assigns the premiumness."""
        return random.choices(
            population=[True, False], weights=[chance, 1 - chance], k=1
        )[0]

    def get_random_impostorness(self, chance):
        """Randomly assigns the impostorness."""
        return random.choices(
            population=[True, False], weights=[chance, 1 - chance], k=1
        )[0]

    def estimate_self_attractiveness(self):
        """Estimates its own attractiveness with an error margin. Agents cannot use
        their actual attractiveness levels for anything. Instead, they use their
        estimations (unless they use a strategy that updates this estimation).
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
        """Returns the agent's ID, attractivenes, and observable attributes."""
        return {
            "id": self.id,
            "attractiveness": self.attractiveness,
            "observable_attributes": self.observable_attributes,
        }

    def get_assessed_candidates(self):
        """Returns the liked or passed candidates' IDs.

        Returns:
            set: A set of seen candidates.
        """
        return self.liked.union(self.passed)

    def assess_candidates(self, candidates_details):
        """Classifies the provided candidates into liked and passed candidates.

        Args:
            candidates_details (list): A list of public candidate details.

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
        """Returns the match happiness (utility) for a given agent.

        Args:
            matched_agent (Agent): An Agent object.

        Returns:
            float: Match happiness.
        """
        if self.is_impostor or (not self.is_impostor and not matched_agent.is_impostor):
            compatibility = self.compatibility_calculator.get_compatibility(
                type_1=[self.observable_attributes + self.hidden_attributes],
                type_2=[
                    matched_agent.observable_attributes
                    + matched_agent.hidden_attributes
                ],
            )

            happiness = (matched_agent.attractiveness * compatibility + 2) ** (0.9) - 1

            if (
                self.gender == "Female"
                and matched_agent.attractiveness - self.estimated_attractiveness > 0
            ):
                discomfort_divisor = (
                    (5 - self.estimated_attractiveness)
                    / (5 * self.estimated_attractiveness)
                ) + (matched_agent.attractiveness - self.estimated_attractiveness) ** (
                    1 / 100
                )
                happiness /= discomfort_divisor
        else:
            happiness = 0
        return happiness

    def get_matched(self, matched_agent):
        """Informs the agent about the match with the provided agent. Each new match
        yields a dimished return to the agent as its number of matches grows.

        Args:
            matched_agent (Agent): An Agent object.

        Returns:
            bool: Returning True indicates the match is acknowledged.
        """
        self.match_count += 1
        self.matched.add(matched_agent.id)
        self.happiness += self.calculate_happiness(matched_agent=matched_agent) * (
            0.999**self.match_count
        )
        self.strategy.match_callback(self, matched_agent)
        return True

    def log_state(self, log_id, logged_variables):
        """Logs the agent's current state with the given log (round) ID and the
        requested log variables.

        Args:
            log_id (int): Round ID that will also uniquely identify the log.
            logged_variables (list): A list of variable names from the agent object.

        Returns:
            bool: Returning True indicates the state is logged.
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
            log_id (int, optional): The round state that will be returned. If None, all
                round logs are returned. Defaults to None.
            variables (list, optional): The set of variables that will be returned. If
                None, all available variables are returned. Defaults to None.

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
