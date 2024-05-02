# Changelog

Notable changes are listed below.

## [0.1.0] - 2024-05-02

### Overview

0.1.0 is the first version published on PyPI.

### Added

- pyproject.toml file.

### Changed

- Rename the package to "catfish-sim" for brevity and to align with conventions, as PyPI prefers shorter package names.

### Removed

- `Strategy` subclass `AdaptiveWeightedMinimal` and Optuna dependency: This strategy was written to make the agent adapt its reported preferences based on its past success, but the preliminary results tests suggested it was not working as intended.