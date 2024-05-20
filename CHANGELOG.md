# Changelog

Notable changes are listed below.

## [0.2.0] - 2024-05-19

### Added

- `DictBasedPreference`: A new preference class that can be used to exactly specify the compatibility score for all attribute values. While `CategoricalPreference` class was able to handle value-compatibility dictionaries, `DictBasedPreference` allows setting a default compatibility score for attribute values that are not specified, and its parameters are more intuitive to use with a dictionary.
- A "getting started" tutorial in the documentation.
- Missing parameter explanations in the docstrings.

### Changed

- Added deal-breaker consideration for `RankedAgentMatcher`: It is now possible to consider one-sided or two-sided deal-breakers while making recommendations.
- Other minor code and documentation improvements, typo fixes.

### Developer notes

- scikit-learn was intended to be dropped from the dependencies. However, due to self-written min-max scaling functions not yielding the exact same results with the one imported from scikit-learn, it was decided to keep scikit-learn.
- `CategoricalPreference` still has the functionality to handle compatibility dictionaries, but this feature may be deprecated in later versions, as there is now a specific class for dictionaries, `DictBasedPreference`.

## [0.1.0] - 2024-05-02

### Overview

0.1.0 is the first version published on PyPI.

### Added

- `pyproject.toml` file.	

### Changed

- Renamed the package to "catfish-sim" for brevity and to align with conventions, as PyPI prefers shorter package names.

### Removed

- `Strategy` subclass `AdaptiveWeightedMinimal` and Optuna dependency: This strategy was written to make the agent adapt its reported preferences based on its past success, but the preliminary test results suggested it was not working as intended.