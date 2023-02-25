import random
import numpy as np
from scipy import stats, optimize


class Strategy:
    """Strategy parent class."""

    def is_interested(self, agent, candidate_details):
        pass

    def match_callback(self, agent, matched_agent):
        pass

    def new_round_callback(self, agent):
        pass


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


class Homophiliac(Strategy):
    """A strategy that favors candidates who are in a specified attractiveness threshold
    with the agent's own estimated attractiveness.
    """

    name = "Homophiliac"

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


class HR(Strategy):
    """A strategy that observes and rejects the first 37% of the candidates that will be
    faced each round, and likes every candidate who is more attractive than the
    observations. This is a crude adaptation of the optimal stopping criteria for the
    secretary problem.
    """

    name = "HR"

    def __init__(self, estimated_daily_candidates=100, default_rejection_threshold=37):
        """Initializes the strategy with the given daily candidate count and the
        rejection threshold.

        Args:
            estimated_daily_candidates (int, optional): The number of candidates the
                agent can face in each round. Defaults to 100.
            default_rejection_threshold (float, optional): The percentile of candidates
                that will be observed and automatically rejected. Defaults to 37.
        """
        self.is_premium = False
        self.estimated_daily_candidates = estimated_daily_candidates
        self.default_rejection_threshold = default_rejection_threshold
        self.observed_values = []

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        if agent.is_premium and not self.is_premium:
            self.is_premium = True
            self.estimated_daily_candidates *= 2
        if len(self.observed_values) >= int(
            self.default_rejection_threshold * 0.01 * self.estimated_daily_candidates
        ):
            if candidate_details["attractiveness"] > max(self.observed_values):
                return True
        else:  # Rejects by default.
            self.observed_values.append(candidate_details["attractiveness"])
            return False

    def new_round_callback(self, agent):
        """A callback function used to notify the strategy about a new round. This is
        used to reset the observed agents for each round.

        Args:
            agent: Not used.
        """
        self.observed_values = []


class AmbitiousHR(Strategy):
    """A strategy that observes and rejects the first 37% of the candidates that will
    be faced each round, and likes every candidate who is more attractive than the
    observations. This is a crude adaptation of the optimal stopping criteria for the
    secretary problem. This strategy constantly updates its observations and they are
    not reset with a new round, making the agent more likely to pass a candidate as the
    time passes.
    """

    name = "Ambitious HR"

    def __init__(self, estimated_daily_candidates=100, default_rejection_threshold=37):
        """Initializes the strategy with the given daily candidate count and the
        rejection threshold.

        Args:
            estimated_daily_candidates (int, optional): The number of candidates the
                agent can face in each round. Defaults to 100.
            default_rejection_threshold (float, optional): The percentile of candidates
                that will be observed and automatically rejected. Defaults to 37.
        """
        self.is_premium = False
        self.estimated_daily_candidates = estimated_daily_candidates
        self.default_rejection_threshold = default_rejection_threshold
        self.observed_values = []

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        if agent.is_premium and not self.is_premium:
            self.is_premium = True
            self.estimated_daily_candidates *= 2
        self.observed_values.append(
            candidate_details["attractiveness"]
        )  # All observations are used.
        if len(self.observed_values) > int(
            self.default_rejection_threshold * 0.01 * self.estimated_daily_candidates
        ):
            if candidate_details["attractiveness"] > max(
                self.observed_values
            ):  # Always wants better even after reaching the threshold.
                return True
        else:  # Rejects by default.
            return False


class Picky(Strategy):
    """A strategy that likes candidates who are better than a specified percentile of
    all previously observed candidates.
    """

    name = "Picky"

    def __init__(self, percentile_threshold=80):
        """Initializes the object with the given percentile threshold to like a
        candidate.

        Args:
            percentile_threshold (float, optional): _description_. Defaults to 80.
        """
        self.percentile_threshold = percentile_threshold
        self.observed_values = []

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        self.observed_values.append(candidate_details["attractiveness"])
        if (
            np.percentile(self.observed_values, self.percentile_threshold)
            <= candidate_details["attractiveness"]
        ):
            return True
        else:
            return False


class SelfReflective(Strategy):
    """A strategy that favors candidates who are in a specified attractiveness threshold
    with the agent's own estimated attractiveness. It is self-reflective as it updates
    the estimated attractiveness after getting a match using the matched agents'
    attractivenesses.
    """

    name = "Self-Reflective"

    def __init__(self, bw_method=0.5, homophily_threshold=[-1.5, 2]):
        """Initializes the strategy with the given homophily threshold.

        Args:
            homophily_threshold (list, optional): A [min, max] homophily threshold used
                to decide whether a candidate is similar enough. The difference between
                the agent's estimated attractiveness and the candidate's attractiveness
                must be within this inclusive threshold to be liked. Defaults to
                [-1.5, 2].
        """
        self.matched_values = [1, 2, 3, 4, 5]  # Some initial values needed.
        self.bw_method = bw_method
        self.homophily_threshold = homophily_threshold
        self.kde = stats.gaussian_kde(
            self.matched_values, bw_method=self.bw_method, weights=None
        )  # In case you need serialization, use dill instead of pickle.
        self.estimated_attractiveness = None
        self.type_probs = {}

    def estimate_self_attractiveness(self, agent):
        """Estimates the agent's attractiveness using the matched agents' attractiveness
        levels. The estimation is based on the assumption that an agent's outcome with a
        candidate (being liked or passed) is less certain when their attractiveness
        levels are close. Therefore, it finds the attractiveness level that yields the
        minimal certainty and updates the estimated attractiveness with it.

        Args:
            agent (Agent): The agent object that has the strategy.

        Returns:
            bool: A boolean value indicating the estimation being complete.
        """

        def get_match_certainty(self, attractiveness):
            """Returns a scalar value indicating the certainty of getting a like or pass
            for a given attractiveness level. This function is used to minimize the
            scalar value and find the agent's estimated attractiveness.

            Args:
                attractiveness (float): Attractiveness level between 1.0 and 5.0.

            Returns:
                float: Scalar certainty value.
            """
            whole_area = self.kde.integrate_box_1d(1.0, 5.0)
            left_area = self.kde.integrate_box_1d(1.0, attractiveness) / whole_area
            right_area = self.kde.integrate_box_1d(attractiveness, 5.0) / whole_area
            return abs(left_area - right_area)

        self.kde = stats.gaussian_kde(
            self.matched_values, bw_method=self.bw_method, weights=None
        )

        optimization = optimize.minimize_scalar(
            lambda attractiveness: get_match_certainty(self, attractiveness),
            method="bounded",
            bounds=(1, 5),
        )
        self.estimated_attractiveness = optimization.x
        agent.estimated_attractivenes = self.estimated_attractiveness
        return True

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        if self.estimated_attractiveness is None:
            self.estimated_attractiveness = self.estimate_self_attractiveness(agent)

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

    def match_callback(self, agent, matched_agent):
        """Called by the agent after getting a match to update the self-evaluation.

        Args:
            agent (Agent): The agent object that has the staretgy.
            matched_agent (Agent): The matched agent object.

        Returns:
            bool: A boolean value indicating acknowledgment.
        """
        self.matched_values.append(matched_agent.attractiveness)
        self.estimate_self_attractiveness(agent)
        return True


class Bayesian(Strategy):
    """BEWARE: THIS STRATEGY IS NOT TESTED PROPERLY AND MAY CHANGE. While it is similar
    to SelfReflective, it calculates and considers the expected reward for a candidate
    while liking or passing an agent.
    """

    name = "Bayesian"

    def __init__(self, bw_method=0.5):
        """Initializes the strategy with the given bw_method for stats.gaussian_kde.

        Args:
            bw_method (float, optional): bw_method parameter that will be passed to the
                stats.gaussian_kde object. Defaults to 0.5.
        """
        self.matched_values = [1, 2, 3, 4, 5]  # Some initial values needed.
        self.bw_method = bw_method
        self.kde = stats.gaussian_kde(
            self.matched_values, bw_method=self.bw_method, weights=None
        )  # In case you need serialization, use dill instead of pickle.
        self.estimated_attractiveness = None
        self.type_probs = {}

    def estimate_self_attractiveness(self, agent):
        """Estimates the agent's attractiveness using the matched agents' attractiveness
        levels. The estimation is based on the assumption that an agent's outcome with a
        candidate (being liked or passed) is less certain when their attractiveness
        levels are close. Therefore, it finds the attractiveness level that yields the
        minimal certainty and updates the estimated attractiveness with it.

        Args:
            agent (Agent): The agent object that has the strategy.

        Returns:
            bool: A boolean value indicating the estimation being complete.
        """

        def get_match_certainty(self, attractiveness):
            """Returns a scalar value indicating the certainty of getting a like or pass
            for a given attractiveness level. This function is used to minimize the
            scalar value and find the agent's estimated attractiveness.

            Args:
                attractiveness (float): Attractiveness level between 1.0 and 5.0.

            Returns:
                float: Scalar certainty value.
            """
            whole_area = self.kde.integrate_box_1d(1.0, 5.0)
            left_area = self.kde.integrate_box_1d(1.0, attractiveness) / whole_area
            right_area = self.kde.integrate_box_1d(attractiveness, 5.0) / whole_area
            return abs(left_area - right_area)

        self.kde = stats.gaussian_kde(
            self.matched_values, bw_method=self.bw_method, weights=None
        )
        optimization = optimize.minimize_scalar(
            lambda attractiveness: get_match_certainty(self, attractiveness),
            method="bounded",
            bounds=(1, 5),
        )
        self.estimated_attractiveness = optimization.x
        agent.estimated_attractivenes = self.estimated_attractiveness
        return True

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): An agent object that will like or pass the candidate.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        if self.estimated_attractiveness is None:
            self.estimated_attractiveness = self.estimate_self_attractiveness(agent)

        whole_area = self.kde.integrate_box_1d(1.0, 5.0)
        left_area = (
            self.kde.integrate_box_1d(1.0, candidate_details["attractiveness"])
            / whole_area
        )
        right_area = (
            self.kde.integrate_box_1d(candidate_details["attractiveness"], 5.0)
            / whole_area
        )

        observable_attributes = ",".join(
            [str(a) for a in candidate_details["observable_attributes"]]
        )
        if observable_attributes not in self.type_probs:
            expected_compatibility = 1
        else:
            expected_compatibility = 0
            total_observations = sum(self.type_probs[observable_attributes].values())
            for hidden_attributes, freq in self.type_probs[
                observable_attributes
            ].items():
                compatibility = agent.compatibility_calculator.get_compatibility(
                    type_1=[agent.observable_attributes + agent.hidden_attributes],
                    type_2=[
                        observable_attributes.split(",") + hidden_attributes.split(",")
                    ],
                )
                expected_compatibility += compatibility * freq / total_observations

        if (
            candidate_details["attractiveness"] * expected_compatibility >= 3 * 1
        ):  # Baseline utility is based on one's own attractiveness.
            return random.choices(
                population=[True, False], weights=[right_area, left_area], k=1
            )[0]
        else:
            return False

    def match_callback(self, agent, matched_agent):
        """Called by the agent after getting a match to update the self-evaluation.

        Args:
            agent (Agent): The agent object that has the staretgy.
            matched_agent (Agent): The matched agent object.

        Returns:
            bool: A boolean value indicating acknowledgment.
        """
        self.matched_values.append(matched_agent.attractiveness)
        observable_attributes = ",".join(
            [str(a) for a in matched_agent.observable_attributes]
        )
        hidden_attributes = ",".join([str(a) for a in matched_agent.hidden_attributes])
        if observable_attributes not in self.type_probs:
            self.type_probs[observable_attributes] = {hidden_attributes: 1}
        elif hidden_attributes not in self.type_probs[observable_attributes]:
            self.type_probs[observable_attributes][hidden_attributes] = 1
        else:
            self.type_probs[observable_attributes][hidden_attributes] += 1
        self.estimate_self_attractiveness(agent)
        return True


class Observant(Strategy):
    """A strategy that observes the seen candidates' and matched candidates'
    attractiveness levels to estimate a candidate's probability to lik the agent. This
    probability is later used to probabilistically like or pass the candidate (a
    candidate with a higher probability to like the agent is liked with a higher
    probability).
    """

    name = "Observant"

    def __init__(self):
        """Initializes the strategy with initial and unbiased distributions of
        observations.
        """
        self.matched_values = [1, 2, 3, 4, 5]  # Some initial values needed.
        self.observed_values = [1, 2, 3, 4, 5]  # Some initial values needed.

    def is_interested(self, agent, candidate_details):
        """Decides whether the given agent is interested in the given candidate.

        Args:
            agent (Agent): Not used.
            candidate_details (dict): Public details of the candidate.

        Returns:
            bool: A boolean value indicating agent's liking (True) or passing (False)
                behavior.
        """
        matched_percentile = stats.percentileofscore(
            self.matched_values, candidate_details["attractiveness"], kind="strict"
        )
        observed_percentile = stats.percentileofscore(
            self.observed_values, candidate_details["attractiveness"], kind="strict"
        )
        like_chance = (matched_percentile + observed_percentile) / 2
        self.observed_values.append(candidate_details["attractiveness"])
        return random.choices(
            population=[True, False], weights=[like_chance, 100 - like_chance], k=1
        )[0]

    def match_callback(self, agent, matched_agent):
        """Adds the matched agent's attractiveness to the distribution of observed
        attractiveness levels.

        Args:
            agent (Agent): Not used.
            matched_agent (Agent): Matched Agent object.

        Returns:
            bool: Returning True indicates the matched agent is added to the
                distribution.
        """
        self.matched_values.append(matched_agent.attractiveness)
        return True
