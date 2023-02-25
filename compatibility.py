class CompatibilityCalculator:
    def __init__(self, attribute_weights, attribute_match_rewards):
        """Initializes the compatibility calculator with the provided attribute weights
        and attribute matching rewards that will be used to calculate the compatibility
        multiplier.

        Args:
            attribute_weights (list): A list of compatibility weights between 0 and 1,
                corresponding to attributes. An example for two equally weighted
                attributes: [0.5, 0.5]
            attribute_match_rewards (list): A list of dictionaries that correspond to
                the reward for each attribute in the case of match (True) or mismatch
                (False). An example for two attributes: [{True: 0.75, False: 1.25},
                {True: 1.25, False: 0.75}]
        """
        self.attribute_weights = attribute_weights
        self.attribute_match_rewards = attribute_match_rewards

    def get_compatibility(self, type_1, type_2):
        """Calculates and returns the compatibility multiplier of two agent types
        (sets of attributes).

        Args:
            type_1 (list): The list of categorical attributes for one agent.
            type_2 (list): The list of categorical attributes for the other agent.

        Returns:
            float: Compatibility multiplier.
        """
        compatibility = 0
        for i in range(len(type_1)):
            compatibility += (
                self.attribute_match_rewards[i][(type_1[i] == type_2[i])]
                * self.attribute_weights[i]
            )

        return compatibility
