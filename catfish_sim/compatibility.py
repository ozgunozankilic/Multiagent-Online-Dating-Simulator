class CompatibilityCalculator:
    def __init__(self):
        pass

    def get_compatibility(self, judger_attributes, judged_attributes):
        """Calculates the compatibility between two agents using two dictionaries of attributes. Since the compatibility
        is calculated from the judging agent's perspective, it is not necessarily symmetrical. Also, when used by agents
        themselves, judger_attributes consists of both reported and hidden attributes, and hidden attributes have the
        precedence over the reported ones to consider the agent's hidden attributes and preferences. judged_attributes
        are always the reported ones. When used by a matcher, both attributes are reported attributes.

        Args:
            judger_attributes (dict): Dictionary of the judging agent's attributes. Consists of both reported and hidden
                attributes where the hidden ones have the precedence when the compatibility is calculated by the agent.
                Otherwise, judger_attributes is a dictionary of reported attributes.
            judged_attributes (dict): Dictionary of the judged agent's reported attributes.

        Returns:
            float: Compatibility score.
        """
        if type(judger_attributes) is dict:
            judger_attributes = [
                judger_attributes[key]
                for key in sorted(judger_attributes.keys(), reverse=True)
            ]
            judged_attributes = [
                judged_attributes[key]
                for key in sorted(judged_attributes.keys(), reverse=True)
            ]

        compatibility = 0
        total_weight = 0
        for i in range(len(judger_attributes)):
            compatibility += (
                judger_attributes[i].preference.evaluate_attribute(
                    judged_attributes[i].value
                )
                * judger_attributes[i].preference.compatibility_weight
            )
            total_weight += judger_attributes[i].preference.compatibility_weight
        compatibility /= total_weight

        return compatibility


class Attribute:
    def __init__(self, name, value, preference):
        """Initializes the Attribute object with the given properties.

        Args:
            name (str): Attribute name.
            value (str|float|int): Attribute value.
            preference (Preference): A Preference object related to the attribute.
        """
        self.name = name
        self.value = value
        self.preference = preference

    def __str__(self):
        """Returns human-readable representation of the attribute.

        Returns:
            str: Human-readable attribute details.
        """
        return f"Attribute(name={self.name}, value={self.value}, preference={self.preference})"

    def __repr__(self):
        """Like __str__, returns human-readable representation of the attribute.

        Returns:
            str: Human-readable attribute details.
        """
        return f"Attribute(name={self.name}, value={self.value}, preference={self.preference})"


class Preference:
    def __init__(
        self,
        preferred_values=None,
        preferred_range=None,
        allowed_values=None,
        allowed_range=None,
        preferred_score=1,
        nonpreferred_score=0,
        distance_sensitive=False,
        compatibility_weight=1,
        compatibility_fn=None,
    ):
        """Initializes the Preference object with the given properties.

        Args:
            preferred_values (list, optional): A list of preferred categorical attributes. Defaults to None.
            preferred_range (list, optional): A list of preferred range, as [min, max], for continuous attributes. Defaults
                to None.
            allowed_values (list, optional): A list of allowed categorical values. Defaults to None.
            allowed_range (list, optional): A list of the allowed range, as [min, max], for continuous attributes.
                Defaults to None.
            preferred_score (float, optional): The attribute-specific compatibility score when the preference is met.
                Defaults to 1.
            nonpreferred_score (float, optional): The attribute-specific compatibility score when the preference is not
                met. Defaults to 0.
            distance_sensitive (bool, optional): Indicates whether the compatibility is mapped according to the allowed
                range and the difference between the closest preferred value and the compared attribute. Only usable for
                continuous attributes. Otherwise, all non-met preferences yield the same non-preferred score. Defaults
                to False.
            compatibility_weight (float, optional): Weight multiplier used to set the attribute preference's weight in the
                overall compatibility. Defaults to 1.
            compatibility_fn (callable, optional): A compatibility function to be used in distance-sensitive (relative)
                compatibility for when the compared attribute does not satisfy the preferred range. Defaults to None.
        """
        self.preferred_values = preferred_values
        if preferred_range and len(preferred_range) == 1:
            preferred_range = preferred_range * 2  # [x] -> [x, x]
        self.preferred_range = preferred_range
        self.preferred_score = preferred_score
        self.nonpreferred_score = nonpreferred_score
        self.allowed_values = allowed_values
        self.allowed_range = allowed_range
        self.distance_sensitive = distance_sensitive
        self.compatibility_weight = compatibility_weight
        if (
            distance_sensitive
            and compatibility_fn is None
            and compatibility_weight
            and self.allowed_range
        ):
            max_diff = abs(self.allowed_range[1] - self.allowed_range[0])
            min_diff = 0
            self.compatibility_fn = lambda a, b: 1 + (nonpreferred_score - 1) / (
                max_diff - min_diff
            ) * abs(a - b)

    def evaluate_attribute(self, value):
        pass


class CategoricalPreference(Preference):
    def __init__(
        self,
        preferred_values,
        allowed_values,
        preferred_score,
        nonpreferred_score,
        compatibility_weight=1,
        ignore_unrecognized=True,
    ):
        """Initializes the categorical preference with the provided properties.

        Args:
            preferred_values (list): A list of preferred categorical attribute values.
            allowed_values (list): A list of all possible categorical attribute values. It is used to throw an error
                with unrecognized values if ignore_unrecognized is set to False.
            preferred_score (float): Compatibility score when the attribute is preferred.
            nonpreferred_score (float): Compatibility score when the attribute is not preferred.
            compatibility_weight (float, optional): Weight multiplier used to set the attribute preference's weight in
                the overall compatibility. Defaults to 1.
            ignore_unrecognized (bool, optional): Specifies whether evaluating a value that is not included in
                allowed_values should be ignored and yield the nonpreferred_score. Defaults to True.
        """
        super().__init__(
            preferred_values=preferred_values,
            allowed_values=allowed_values,
            preferred_score=preferred_score,
            nonpreferred_score=nonpreferred_score,
            compatibility_weight=compatibility_weight,
        )
        self.ignore_unrecognized = ignore_unrecognized

    def evaluate_attribute(self, value):
        """Evaluates a candidate's attribute, comparing it with the preferred values and returning an
        attribute-specific compatibility score.

        Args:
            value (str|float|int): Candidate's attribute value.

        Returns:
            float: Attribute-specific compatibility score.
        """
        if isinstance(self.preferred_score, dict) and value in self.preferred_score:
            return self.preferred_score[value]
        elif value in self.preferred_values:
            return self.preferred_score
        elif self.ignore_unrecognized:
            return self.nonpreferred_score
        else:
            raise ValueError(
                f'The evaluated value "{value}" is not recognized (it is not listed in allowed_values).'
            )

    def __str__(self):
        """Returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"CategoricalPreference(\n\tpreferred_values={self.preferred_values}, \n\tpreferred_score={self.preferred_score}, \n\tnonpreferred_score={self.nonpreferred_score}\n)"

    def __repr__(self):
        """Like __str__, returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"CategoricalPreference(\n\tpreferred_values={self.preferred_values}, \n\tpreferred_score={self.preferred_score}, \n\tnonpreferred_score={self.nonpreferred_score}\n)"


class NumericalPreference(Preference):
    def __init__(
        self,
        preferred_range,
        allowed_range,
        preferred_score,
        nonpreferred_score,
        distance_sensitive=None,
        compatibility_weight=1,
        compatibility_fn=None,
    ):
        """Initializes the numerical preference with the provided properties.

        Args:
            preferred_range (list): A list of preferred numerical attribute range, as [min, max].
            allowed_range (list): A list of the possible numerical attribute range, as [min, max].
            preferred_score (float): Compatibility score when the attribute is preferred.
            nonpreferred_score (float): Compatibility score when the attribute is not preferred.
            distance_sensitive (bool, optional): Indicates whether the compatibility is mapped according to the allowed
                range and the difference between the closest preferred value and the compared attribute. Only usable for
                continuous attributes. Otherwise, all non-met preferences yield the same non-preferred score. Defaults
                to False.
            compatibility_weight (float, optional): Weight multiplier used to set the attribute preference's weight in
                the overall compatibility. Defaults to 1.
            compatibility_fn (callable, optional): A compatibility function to be used in distance-sensitive (relative)
                compatibility for when the compared attribute does not satisfy the preferred range. Defaults to None.
        """
        super().__init__(
            preferred_range=preferred_range,
            allowed_range=allowed_range,
            preferred_score=preferred_score,
            nonpreferred_score=nonpreferred_score,
            distance_sensitive=distance_sensitive,
            compatibility_weight=compatibility_weight,
            compatibility_fn=compatibility_fn,
        )

    def evaluate_attribute(self, value):
        """Evaluates a candidate's attribute, comparing it with the preferred value range and returning an
        attribute-specific compatibility score.

        Args:
            value (float): Candidate's attribute value.

        Returns:
            float: Attribute-specific compatibility score.
        """
        if self.preferred_range[0] <= value <= self.preferred_range[1]:
            return self.preferred_score
        elif self.distance_sensitive is False:
            return self.nonpreferred_score
        else:  # relative loss based on the preferred range
            if value < self.preferred_range[0]:
                return self.compatibility_fn(value, self.preferred_range[0])
            else:
                return self.compatibility_fn(value, self.preferred_range[1])

    def __str__(self):
        """Returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"NumericalPreference(\n\tpreferred_range={self.preferred_range}, \n\tpreferred_score={self.preferred_score}, \n\tnonpreferred_score={self.nonpreferred_score}, \n\tdistance_sensitive={self.distance_sensitive}, \n\tcompatibility_weight={self.compatibility_weight}, \n\tcompatibility_fn={self.compatibility_fn if hasattr(self, 'compatibility_fn') else None}\n)"

    def __repr__(self):
        """Like __str__, returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"NumericalPreference(\n\tpreferred_range={self.preferred_range}, \n\tpreferred_score={self.preferred_score}, \n\tnonpreferred_score={self.nonpreferred_score}, \n\tdistance_sensitive={self.distance_sensitive}, \n\tcompatibility_weight={self.compatibility_weight}, \n\tcompatibility_fn={self.compatibility_fn if hasattr(self, 'compatibility_fn') else None}\n)"


class DictBasedPreference(Preference):
    def __init__(
        self,
        compatibility_dict,
        default_value=None,
        compatibility_weight=1,
    ):
        """Initializes the dictionary-based preference with the provided properties.

        Args:
            compatibility_dict (dict): A dictionary that maps attribute values to compatibility scores.
            default_value (Any, optional): A default compatibility value to be returned when the evaluated value does not exist
                in the dictionary. Defaults to None.
            compatibility_weight (float, optional): Weight multiplier used to set the attribute preference's weight in
                the overall compatibility. Defaults to 1.
        """
        self.compatibility_dict = compatibility_dict
        self.default_value = default_value
        self.compatibility_weight = compatibility_weight

    def evaluate_attribute(self, value):
        """Evaluates a candidate's attribute, comparing it with the preferred value range and returning an
        attribute-specific compatibility score.

        Args:
            value (float): Candidate's attribute value.

        Returns:
            float: Attribute-specific compatibility score.
        """
        return self.compatibility_dict.get(value, self.default_value)

    def __str__(self):
        """Returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"DictPreference(\n\tcompatibility_dict={self.compatibility_dict}, \n\tdefault_value={self.default_value}, \n\tcompatibility_weight={self.compatibility_weight})"

    def __repr__(self):
        """Like __str__, returns human-readable representation of the preference.

        Returns:
            str: Human-readable preference details.
        """
        return f"DictPreference(\n\tcompatibility_dict={self.compatibility_dict}, \n\tdefault_value={self.default_value}, \n\tcompatibility_weight={self.compatibility_weight})"
