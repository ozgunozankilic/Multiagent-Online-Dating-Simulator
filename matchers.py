import random
import scipy.sparse
import numpy as np
from trueskill import Rating, rate_1vs1


class AgentMatcher:
    """This is the base AgentMatcher class that is extended by specific matchers."""

    def get_agents_by_ids(self, ids):
        """Returns a list of Agent objects that correspond to the provided list of IDs.

        Args:
            ids (list): A list of agent IDs.

        Returns:
            list: A list of Agent objects.
        """
        return [agent for agent in self.agents if agent.id in ids]

    def get_agent_by_id(self, id):
        """Returns the agent that corresponds to the provided ID.

        Args:
            id (int): Agent ID.

        Returns:
            Agent: An Agent object.
        """
        return self.agents[id]

    def get_available_candidates(self, agent):
        """Retrieves available candidate agents that will be recommended to the agent."""
        pass

    def process_likes(self, likes):
        """Processes likes by updating the matching matrix."""
        pass

    def process_passes(self, passes):
        """Processes passes by updating the matching matrix."""
        pass

    def process_matches(self):
        """Processes matches by informing the agents about their match details."""
        pass

    def run_new_round(self):
        """Runs the simulation for a new round. Each agent is notified of the new round,
        their allowance is replenished, new agents are recommended, and agent responses
        are saved. After this is done for all agents, likes, passes, and matches are
        processed, and the agents are logged.
        """
        pass

    def eliminate_bottom(self, bottom_threshold):
        """Eliminates the provided bottom percentile (inclusive) of agents from the
        simulation. Eliminated agents cannot interact with the system and they are not
        shown to other agents anymore.
        """
        pass

    def get_info(self):
        """Returns details of the matcher."""
        pass


class RandomAgentMatcher(AgentMatcher):
    """This matcher randomly recommends unseen agents to each agent based on their
    remaining like allowance. Agents' IDs must start from 0 and be sequential due to
    being used as indices.
    """

    name = "Random Agent Matcher"

    def __init__(self, agents, recommendation_limit, compatibility_calculator):
        """Initiates the matcher with provided parameters.

        Args:
            agents (list): A list of Agent objects.
            recommendation_limit (int): Round-wise maximum number of agents that can be
                recommended.
            compatibility_calculator (CompatibilityCalculator): A
                CompatibilityCalculator object that will be used to calculate the agent
                compatibilities.
        """
        self.agents = agents
        # Agents' IDs must start from 0 and be sequential due to being used as indices.
        self.agent_ids = set([agent.id for agent in self.agents])
        self.round = 0
        self.matrix = scipy.sparse.dok_matrix(
            (len(self.agent_ids), len(self.agent_ids)), dtype=int
        )
        self.recommendation_limit = recommendation_limit
        self.compatibility_calculator = compatibility_calculator
        self.waiting_matches = set()
        self.eliminated_agents = set()

    def get_info(self):
        """Returns details of the matcher.

        Returns:
            dict: Name, agents, rounds, and the recommendation limit of the matcher.
        """
        return {
            "name": self.name,
            "agents": len(self.agent_ids),
            "rounds": self.round,
            "recommendation_limit": self.recommendation_limit,
        }

    def get_available_candidates(self, agent):
        """Returns all candidates who were not seen by the agent.

        Args:
            agent (list): A list of Agent objects.

        Returns:
            set: A set of available agent IDs.
        """
        return (self.agent_ids.difference(agent.get_assessed_candidates())).difference(
            set([agent.id]).union(self.eliminated_agents)
        )

    def process_likes(self, likes):
        """Processes likes by updating the matching matrix.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        for agent_id, likes in likes.items():
            for liked_id in likes:
                self.matrix[agent_id, liked_id] = 1
                # If the liked agent had liked back, a new match is registered.
                if self.matrix[liked_id, agent_id] == 1:
                    self.waiting_matches.add(tuple(sorted([agent_id, liked_id])))
        return True

    def process_passes(self, passes):
        """Processes passes by updating the matching matrix.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        for agent_id, passed in passes.items():
            for passed_id in passed:
                self.matrix[agent_id, passed_id] = -1
        return True

    def process_matches(self):
        """Processes matches by informing the agents about their match details.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        for agent_id_1, agent_id_2 in self.waiting_matches:
            agent_1 = super().get_agent_by_id(agent_id_1)
            agent_2 = super().get_agent_by_id(agent_id_2)
            agent_1.get_matched(agent_2)
            agent_2.get_matched(agent_1)
        self.waiting_matches = set()
        return True

    def run_new_round(self):
        """Runs the simulation for a new round. Each agent is notified of the new round,
        their allowance is replenished, new agents are recommended, and agent responses
        are saved. After this is done for all agents, likes, passes, and matches are
        processed, and the agents are logged.

        Returns:
            bool: Returning True indicates the round is complete.
        """
        self.round += 1
        round_likes = {}
        round_passes = {}
        for agent in self.agents:
            if agent.id in self.eliminated_agents:
                continue
            agent.strategy.new_round_callback(agent)
            agent.remaining_likes = agent.like_allowance
            available_candidates = self.get_available_candidates(agent)
            n_recommend = min(self.recommendation_limit, len(available_candidates))
            recommended_agent_ids = random.sample(available_candidates, n_recommend)
            candidates_details = [
                agent.get_public_details()
                for agent in super().get_agents_by_ids(recommended_agent_ids)
            ]
            assessments = agent.assess_candidates(candidates_details)
            round_likes[agent.id] = assessments["liked"]
            round_passes[agent.id] = assessments["passed"]

        self.process_likes(likes=round_likes)
        self.process_passes(passes=round_passes)
        self.process_matches()
        self.log_agents()

    def eliminate_bottom(self, bottom_threshold, verbose=False):
        """Eliminates the bottom percentile based on happiness. Eliminated agents cannot
        interact with the system and they are not shown to other agents anymore.

        Args:
            bottom_threshold (float): Inclusive percentile threshold that will be used
                to eliminate agents.
            verbose (bool, optional): Indicates whether the elimination details (round
                and number of remaining agents) will be printed. Defaults to False.

        Returns:
            bool: Returning True indicates the elimination is complete.
        """
        threshold_value = np.percentile(
            [
                agent.happiness
                for agent in self.agents
                if agent.id not in self.eliminated_agents
            ],
            bottom_threshold,
        )
        for agent in self.agents:
            if agent.happiness <= threshold_value:
                self.eliminated_agents.add(agent.id)
        if verbose:
            print(
                f"Round {self.round}: {len(self.agent_ids.difference(self.eliminated_agents))} users are remaining.",
                flush=True,
            )
        return True

    def log_agents(self):
        for agent in self.agents:
            agent.log_state(
                log_id=self.round, logged_variables=["match_count", "happiness"]
            )
        return True


class RankedAgentMatcher(AgentMatcher):
    """A matcher that ranks agents based on their interactions with other agents using
    TrueSkill, a Bayesian rating system. It can eliminate agents based on their
    happiness levels or ratings. Since TrueSkill works with wins, losses, and draws,
    likes and passes are interpreted as 1v1 matches. By default, liking an agent means
    losing a match to them, while passing an agent means winning the match. However, the
    matcher is flexible and it is possible to change how these interactions are
    interpreted by the matcher. Agents' IDs must start from 0 and be sequential due to
    being used as indices.
    """

    name = "Ranked Agent Matcher"

    def __init__(
        self,
        agents,
        recommendation_limit,
        compatibility_calculator,
        strict_recommendations=True,
        likes_as_draws=False,
        rank_passes=True,
        rank_pass_from_lower=True,
        rank_pass_from_higher=True,
    ):
        """Initiates the matcher with provided parameters.

        Args:
            agents (list): A list of Agent objects.
            recommendation_limit (int): Round-wise maximum number of agents that can be
                recommended.
            compatibility_calculator (CompatibilityCalculator): A
                CompatibilityCalculator object that will be used to calculate the agent
                compatibilities.
            strict_recommendations (bool, optional): Determines whether the recommended
                agents have strictly the closest ratings to the agent. Otherwise,
                recommendation_limit agents are sampled from the closest
                recommendation_limit * 1.5 agents. Defaults to True.
            likes_as_draws (bool, optional): Indicates whether liking someone will be
                treated as a draw in the rating system. Otherwise, it counts as a loss
                to the liked agent. Defaults to False.
            rank_passes (bool, optional): Indicates whether passes are considered in the
                rating system. Defaults to True.
            rank_pass_from_lower (bool, optional): Indicates whether a lower-rating
                agent's pass affects the higher-rating agent's rating. Defaults to True.
            rank_pass_from_higher (bool, optional): Indicates whether a higher-rating
                agent's pass affects the lower-rating agent's rating. Defaults to True.
        """
        self.agents = agents
        # Agents' IDs must start from 0 and be sequential due to being used as indices.
        self.agent_ids = set([agent.id for agent in self.agents])
        self.round = 0
        self.matrix = scipy.sparse.dok_matrix(
            (len(self.agent_ids), len(self.agent_ids)), dtype=int
        )
        self.ratings = [Rating()] * len(self.agent_ids)
        self.recommendation_limit = recommendation_limit
        self.compatibility_calculator = compatibility_calculator
        self.waiting_matches = set()
        self.eliminated_agents = set()
        self.rating_logs = {agent_id: {} for agent_id in self.agent_ids}
        self.strict_recommendations = strict_recommendations
        self.likes_as_draws = likes_as_draws
        self.rank_passes = rank_passes
        self.rank_pass_from_lower = rank_pass_from_lower
        self.rank_pass_from_higher = rank_pass_from_higher

    def get_info(self):
        """Returns details of the matcher.

        Returns:
            dict: Name, agents, rounds, and the recommendation limit of the matcher.
        """
        return {
            "name": self.name,
            "agents": len(self.agent_ids),
            "rounds": self.round,
            "recommendation_limit": self.recommendation_limit,
        }

    def get_available_candidates(self, agent):
        """Retrieves previously unseen candidates with close ratings for a specific
        agent.

        Args:
            agent (Agent): An Agent object.

        Returns:
            list: Agent IDs that will be recommended.
        """
        all_available = (
            self.agent_ids.difference(agent.get_assessed_candidates())
        ).difference(set([agent.id]).union(self.eliminated_agents))
        agent_rating = self.ratings[agent.id].mu
        sorted_ratings = {
            i: abs(rating.mu - agent_rating)
            for i, rating in enumerate(self.ratings)
            if i in all_available
        }
        sorted_ratings = dict(
            sorted(sorted_ratings.items(), key=lambda x: x[1], reverse=True)
        )
        if self.strict_recommendations:
            return list(sorted_ratings.keys())[
                : min(len(sorted_ratings), self.recommendation_limit)
            ]
        else:
            return random.sample(
                list(sorted_ratings.keys())[
                    : min(len(sorted_ratings), int(self.recommendation_limit * 1.5))
                ],
                min(len(sorted_ratings), self.recommendation_limit),
            )

    def process_likes(self, likes):
        """Processes likes by updating the matching matrix.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        liker_agents = list(likes.keys())
        random.shuffle(liker_agents)
        for agent_id in liker_agents:
            liked = list(likes[agent_id])
            random.shuffle(liked)
            for liked_id in liked:
                self.matrix[agent_id, liked_id] = 1
                liked_rating, agent_rating = rate_1vs1(
                    self.ratings[liked_id],
                    self.ratings[agent_id],
                    drawn=self.likes_as_draws,
                )
                self.ratings[liked_id] = liked_rating
                self.ratings[agent_id] = agent_rating
                # If the liked agent had liked back, a new match is registered.
                if self.matrix[liked_id, agent_id] == 1:
                    self.waiting_matches.add(tuple(sorted([agent_id, liked_id])))
        return True

    def process_passes(self, passes):
        """Processes passes by updating the matching matrix.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        passing_agents = list(passes.keys())
        random.shuffle(passing_agents)
        for agent_id in passing_agents:
            passed = list(passes[agent_id])
            random.shuffle(passed)
            for passed_id in passed:
                self.matrix[agent_id, passed_id] = -1
                if self.rank_passes:
                    if (
                        not self.rank_pass_from_lower
                        and self.ratings[agent_id] < self.ratings[passed_id]
                    ):
                        continue
                    elif (
                        not self.rank_pass_from_higher
                        and self.ratings[agent_id] > self.ratings[passed_id]
                    ):
                        continue

                    agent_rating, passed_rating = rate_1vs1(
                        self.ratings[agent_id], self.ratings[passed_id]
                    )
                    self.ratings[passed_id] = passed_rating
                    self.ratings[agent_id] = agent_rating
        return True

    def process_matches(self):
        """Processes matches by informing the agents about their match details.

        Returns:
            bool: Returning True indicates the process is complete.
        """
        for agent_id_1, agent_id_2 in self.waiting_matches:
            agent_1 = super().get_agent_by_id(agent_id_1)
            agent_2 = super().get_agent_by_id(agent_id_2)
            agent_1.get_matched(agent_2)
            agent_2.get_matched(agent_1)
        self.waiting_matches = set()
        return True

    def run_new_round(self):
        """Runs the simulation for a new round. Each agent is notified of the new round,
        their allowance is replenished, new agents are recommended, and agent responses
        are saved. After this is done for all agents, likes, passes, and matches are
        processed, and the agents are logged.

        Returns:
            bool: Returning True indicates the round is complete.
        """
        self.round += 1
        round_likes = {}
        round_passes = {}
        for agent in self.agents:
            if agent.id in self.eliminated_agents:
                continue
            agent.strategy.new_round_callback(agent)
            agent.remaining_likes = agent.like_allowance
            available_candidates = self.get_available_candidates(agent)
            n_recommend = min(self.recommendation_limit, len(available_candidates))
            recommended_agent_ids = random.sample(available_candidates, n_recommend)
            candidates_details = [
                agent.get_public_details()
                for agent in super().get_agents_by_ids(recommended_agent_ids)
            ]
            assessments = agent.assess_candidates(candidates_details)
            round_likes[agent.id] = assessments["liked"]
            round_passes[agent.id] = assessments["passed"]

        self.process_likes(likes=round_likes)
        self.process_passes(passes=round_passes)
        self.process_matches()
        self.log_agents()
        return True

    def eliminate_bottom(self, bottom_threshold, using_ratings=False, verbose=False):
        """Eliminates the bottom percentile based on happiness or rating. Eliminated
        agents cannot interact with the system and they are not shown to other agents
        anymore.

        Args:
            bottom_threshold (float): Inclusive percentile threshold that will be used
                to eliminate agents.
            using_ratings (bool, optional): Indicates whether the elimination is handled
                based on ratings. Defaults to False.
            verbose (bool, optional): Indicates whether the elimination details (round
                and number of remaining agents) will be printed. Defaults to False.

        Returns:
            bool: Returning True indicates the elimination is complete.
        """
        if using_ratings:
            threshold_value = np.percentile(
                [
                    self.ratings[agent_id].mu
                    for agent_id in self.agent_ids
                    if agent_id not in self.eliminated_agents
                ],
                bottom_threshold,
            )
            for agent_id in self.agent_ids:
                if self.ratings[agent_id].mu <= threshold_value:
                    self.eliminated_agents.add(agent_id)
        else:
            threshold_value = np.percentile(
                [
                    agent.happiness
                    for agent in self.agents
                    if agent.id not in self.eliminated_agents
                ],
                bottom_threshold,
            )
            for agent in self.agents:
                if agent.happiness <= threshold_value:
                    self.eliminated_agents.add(agent.id)
        if verbose:
            print(
                f"Round {self.round}: {len(self.agent_ids.difference(self.eliminated_agents))} users are remaining.",
                flush=True,
            )
        return True

    def log_agents(self):
        """Logs agents' current state.

        Returns:
            bool: Returns True when the process is complete.
        """
        for agent in self.agents:
            agent.log_state(
                log_id=self.round, logged_variables=["match_count", "happiness"]
            )
            self.rating_logs[agent.id][self.round] = self.ratings[agent.id]
        return True
