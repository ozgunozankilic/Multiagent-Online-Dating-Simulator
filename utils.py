import random
import pandas as pd


def get_random_strategy(strategies, strategy_weights=None):
    """Chooses, instantiates, and returns a random strategy from the provided strategies.

    Args:
        strategies (list): A list of strategy classes.
        strategy_weights (list, optional): A list of strategy weights for sampling. Defaults to None.

    Returns:
        An instantiated strategy object.
    """
    return random.choices(strategies, weights=strategy_weights, k=1)[0]()


def get_agent_stats(agents):
    """Retrieves the provided agents' current status.

    Args:
        agents (list): A list of agent objects.

    Returns:
        DataFrame: A pandas DataFrame with agents' status.
    """
    stats = []
    for agent in agents:
        stat = {
            "ID": agent.id,
            "STRATEGY": agent.strategy.name,
            "GENDER": agent.gender,
            "ATTRACTIVENESS": agent.attractiveness,
            "EST_ATTRACTIVENESS": agent.estimated_attractiveness,
            "PREMIUM": agent.is_premium,
            "IMPOSTOR": agent.is_impostor,
            "ATTRIBUTES": agent.observable_attributes + agent.hidden_attributes,
            "LIKED": len(agent.liked),
            "PASSED": len(agent.passed),
            "MATCHED": agent.match_count,
            "HAPPINESS": agent.happiness,
        }
        stats.append(stat)
    return pd.DataFrame(stats)
