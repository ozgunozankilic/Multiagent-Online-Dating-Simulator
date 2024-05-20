import random
import pandas as pd
import scipy.stats
import math
import os

LLCP2022_AGE_GROUP_RANGE = [1, 13]
LLCP2022_HEIGHT_RANGE = [91, 241]
LLCP2022_BMI_GROUP_RANGE = [1, 4]
LLCP2022_DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data/LLCP2022_sex_age_height_bmi.csv")
)


def get_random_gender(male_weight=0.72):
    """Samples gender.

    Args:
        male_weight (float, optional): Male gender probability. Defaults to 0.72 (based on online dating populations).

    Returns:
        _type_: _description_
    """
    return random.choices(
        ["Male", "Female"], weights=[male_weight, (1 - male_weight)], k=1
    )[0]


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
            "ATTRACTIVENESS": agent.attractiveness,
            "EST_ATTRACTIVENESS": agent.estimated_attractiveness,
            "REPORTED_ATTRIBUTES": agent.reported_attributes,
            "HIDDEN_ATTRIBUTES": agent.hidden_attributes,
            "LIKED": len(agent.liked),
            "PASSED": len(agent.passed),
            "MATCHED": agent.match_count,
            "HAPPINESS": agent.happiness,
        }
        stats.append(stat)
    return pd.DataFrame(stats)


def sample_bmi_from_sex_age(sex, age_group):
    """Samples a weight group based on the provided sex and age group based on the data. Based on LLCP2022:
    https://www.cdc.gov/brfss/annual_data/annual_2022.html

    Args:
        sex (str): "Male" or "Female."
        age_group (str|int): Age group ID (1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54,
            8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80+).

    Returns:
        int: BMI group ID (1: Underweight, 2: Normal weight, 3: Overweight, 4: Obesity)
    """
    age = str(age_group)

    sex_age_bmi = {
        "Male": {
            "1": {  # 18-24
                "1": 0.04840228968915296,
                "2": 0.47235707557423373,
                "3": 0.27845808274762696,
                "4": 0.2007825519889863,
            },
            "2": {  # 25-29
                "1": 0.023654697954952408,
                "2": 0.3539722929035906,
                "3": 0.35793044953350295,
                "4": 0.264442559607954,
            },
            "3": {  # 30-34
                "1": 0.012286689419795221,
                "2": 0.29197952218430034,
                "3": 0.38515358361774743,
                "4": 0.310580204778157,
            },
            "4": {  # 35-39
                "1": 0.009686743223185688,
                "2": 0.2456839309428951,
                "3": 0.3991875634716038,
                "4": 0.34544176236231544,
            },
            "5": {  # 40-44
                "1": 0.006257153758107593,
                "2": 0.21404044257916827,
                "3": 0.41091186570011445,
                "4": 0.3687905379626097,
            },
            "6": {  # 45-49
                "1": 0.006302353410450738,
                "2": 0.18994814519345832,
                "3": 0.40095731950538493,
                "4": 0.40279218189070604,
            },
            "7": {  # 50-54
                "1": 0.007841518778373916,
                "2": 0.16962443252166737,
                "3": 0.412573944146375,
                "4": 0.4099601045535837,
            },
            "8": {  # 55-59
                "1": 0.005810718921926191,
                "2": 0.17765964022995612,
                "3": 0.41045929405946713,
                "4": 0.40607034678865056,
            },
            "9": {  # 60-64
                "1": 0.008853730092204526,
                "2": 0.19975901089689857,
                "3": 0.41733025984911987,
                "4": 0.37405699916177704,
            },
            "10": {  # 65-69
                "1": 0.007913389649480532,
                "2": 0.20919506748227984,
                "3": 0.4301388484318866,
                "4": 0.35275269443635304,
            },
            "11": {  # 70-74
                "1": 0.008408219034105838,
                "2": 0.23422145146881077,
                "3": 0.4316569446634085,
                "4": 0.3257133848336749,
            },
            "12": {  # 75-79
                "1": 0.007979312892500923,
                "2": 0.2574067233099372,
                "3": 0.4494274104174363,
                "4": 0.2851865533801256,
            },
            "13": {  # 80+
                "1": 0.012981298129812982,
                "2": 0.3428676200953429,
                "3": 0.44620462046204623,
                "4": 0.19794646131279794,
            },
        },
        "Female": {
            "1": {  # 18-24
                "1": 0.04786969907637303,
                "2": 0.48912503724302314,
                "3": 0.23408481477803159,
                "4": 0.22892044890257224,
            },
            "2": {  # 25-29
                "1": 0.02436781609195402,
                "2": 0.38114942528735635,
                "3": 0.26586206896551723,
                "4": 0.32862068965517244,
            },
            "3": {  # 30-34
                "1": 0.017873853640276997,
                "2": 0.3429721130451057,
                "3": 0.27063447501403703,
                "4": 0.3685195583005802,
            },
            "4": {  # 35-39
                "1": 0.016477272727272726,
                "2": 0.32987012987012987,
                "3": 0.2814935064935065,
                "4": 0.3721590909090909,
            },
            "5": {  # 40-44
                "1": 0.012779552715654952,
                "2": 0.30277137974589496,
                "3": 0.2917750204324244,
                "4": 0.3926740471060257,
            },
            "6": {  # 45-49
                "1": 0.011997832649585882,
                "2": 0.2928245220218283,
                "3": 0.2927471166498955,
                "4": 0.4024305286786903,
            },
            "7": {  # 50-54
                "1": 0.012770074169622702,
                "2": 0.28068365043534343,
                "3": 0.29532408900354723,
                "4": 0.4112221863914866,
            },
            "8": {  # 55-59
                "1": 0.015816598542740357,
                "2": 0.28837154197026243,
                "3": 0.3054321426455779,
                "4": 0.39037971684141937,
            },
            "9": {  # 60-64
                "1": 0.01994965088110958,
                "2": 0.30028974492946375,
                "3": 0.3140170047024177,
                "4": 0.36574359948700896,
            },
            "10": {  # 65-69
                "1": 0.020012620571531598,
                "2": 0.3116830433606779,
                "3": 0.31393671684846297,
                "4": 0.3543676192193275,
            },
            "11": {  # 70-74
                "1": 0.021507459794613448,
                "2": 0.3193664018601046,
                "3": 0.32454950590970744,
                "4": 0.3345766324355745,
            },
            "12": {  # 75-79
                "1": 0.025180295807358515,
                "2": 0.34842928737318174,
                "3": 0.33944505561667276,
                "4": 0.28694536120278696,
            },
            "13": {  # 80+
                "1": 0.03670282875392882,
                "2": 0.4369360235222549,
                "3": 0.3268275372604684,
                "4": 0.19953361046334786,
            },
        },
    }

    bmi_probs = sex_age_bmi[sex][age]
    return int(
        random.choices(
            population=list(bmi_probs.keys()), weights=bmi_probs.values(), k=1
        )[0]
    )


def sample_age_from_sex(sex, consider_dating_population=True):
    """Samples age from sex (based on LLCP2022: https://www.cdc.gov/brfss/annual_data/annual_2022.html) and optionally
    calibrates the probabilities according to online dating age penetration distribution (based on
    https://www.pewresearch.org/short-reads/2023/02/02/key-findings-about-online-dating-in-the-u-s/).

    Args:
        sex (str): "Male" or "Female."
        consider_dating_population (bool, optional): Indicates whether the age probabilities are calibrated using the
            online dating distributions. Defaults to True.

    Returns:
        int: Age group ID (1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54, 8: 55-59, 9: 60-64,
            10: 65-69, 11: 70-74, 12: 75-79, 13: 80+).
    """

    sex_age = {
        "Male": {
            "1": 0.07398741365778994,
            "2": 0.05641430611817123,
            "3": 0.06279522089469297,
            "4": 0.06805496653553862,
            "5": 0.07010719351866747,
            "6": 0.06602223814607372,
            "7": 0.07659535056034084,
            "8": 0.08438503872908167,
            "9": 0.0991113515937663,
            "10": 0.10595535796980643,
            "11": 0.09738572605450832,
            "12": 0.06916638637438274,
            "13": 0.07001944984717977,
        },
        "Female": {
            "1": 0.05094192542549045,
            "2": 0.045112814516478285,
            "3": 0.055974189077995756,
            "4": 0.06307652332077433,
            "5": 0.06738556147416742,
            "6": 0.06490407518080638,
            "7": 0.0776536312849162,
            "8": 0.0844917933393963,
            "9": 0.10471179247325797,
            "10": 0.1098393313412152,
            "11": 0.10174526871941449,
            "12": 0.07937724654627344,
            "13": 0.09478584729981378,
        },
    }

    online_dating_weights = {
        "1": 0.53,
        "2": 0.53,
        "3": 0.37,
        "4": 0.37,
        "5": 0.37,
        "6": 0.37,
        "7": 0.20,
        "8": 0.20,
        "9": 0.20,
        "10": 0.13,
        "11": 0.13,
        "12": 0.13,
        "13": 0.13,
    }

    age_probs = sex_age[sex]
    if consider_dating_population:
        for age in age_probs.keys():
            age_probs[age] *= online_dating_weights[age]

    return int(
        random.choices(
            population=list(age_probs.keys()), weights=age_probs.values(), k=1
        )[0]
    )


def sample_height_from_sex_age(sex, age_group):
    """Samples height from the provided sex and age group (based on LLCP2022:
    https://www.cdc.gov/brfss/annual_data/annual_2022.html).

    Args:
        sex (str): "Male" or "Female."
        age_group (str|int): Age group ID (1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54,
            8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80+).

    Returns:
        int: Rounded sampled height.
    """
    if sex == "Male":
        sex = 1
    else:  # Female
        sex = 2
    age = int(age_group)

    dataset = LLCP2022_DATA

    kde = scipy.stats.gaussian_kde(
        dataset[
            (dataset["_SEX"] == sex)
            & (dataset["_AGEG5YR"] == age)
            & (~dataset["HTM4"].isnull())
        ]["HTM4"].tolist()
    )
    return int(kde.resample(1)[0][0])


def sample_age_preference(sex, age_group, allowed_diff=3):
    """Samples age preference for a given sex and age group (based on the 2017 US Current Population Survey:
    https://www2.census.gov/programs-surveys/demo/tables/families/2017/cps-2017/tabfg3-all.xls), along with the
    user-provided allowed maximum difference using age group IDs.

    Args:
        sex (str): "Male" or "Female."
        age_group (str|int): Age group ID (1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54,
            8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80+).
        allowed_diff (int, optional): Allowed maximum difference using age group IDs. Defaults to 3.

    Returns:
        list: Age preference range (minimum, maximum).
    """

    observed_differences = {
        "5": 0.01,  # Husband 20+ years older than wife
        "4": 0.014,  # Husband 15-19 years older than wife
        "3": 0.047,  # Husband 10-14 years older than wife
        "2": 0.115,  # Husband 6-9 years older than wife
        "1": 0.122,  # Husband 4-5 years older than wife
        "0": 0.617,  # Husband 2-3 years older than wife / Husband and wife within 1 year / Wife 2-3 years older than husband
        "-1": 0.032,  # Wife 4-5 years older than husband
        "-2": 0.027,  # Wife 6-9 years older than husband
        "-3": 0.009,  # Wife 10-14 years older than husband
        "-4": 0.003,  # Wife 15-19 years older than husband
        "-5": 0.004,  # Wife 20+ years older than husband
    }

    age = age_group

    random_diff = int(
        random.choices(
            population=list(observed_differences.keys()),
            weights=observed_differences.values(),
            k=1,
        )[0]
    )
    if random_diff == 0:
        preference = [
            max(1, math.ceil(age - allowed_diff / 2)),
            min(13, math.floor(age + allowed_diff / 2)),
        ]
        return preference
    elif random_diff > 0:  # Older-male
        if sex == "Male":
            pref_min = max(1, age - random_diff)
            pref_max = max(age, pref_min + allowed_diff)
        else:  # Female
            pref_max = min(13, age + random_diff)
            pref_min = min(age, pref_max - allowed_diff)
    else:  # Older-female
        if sex == "Male":
            pref_max = min(13, age - random_diff)
            pref_min = min(age, pref_max - allowed_diff)
        else:  # Female
            pref_min = max(1, age + random_diff)
            pref_max = max(age, pref_min + allowed_diff)

    preference = [pref_min, pref_max]
    return preference


def get_height_preference(gender, height):
    """Deterministically obtains a height preference range for a given gender and height. Uses the height range from
    LLCP2022 data.

    Args:
        gender (str): Agent gender.
        height (int): Agent height.

    Returns:
        list: Height preference range, as [min, max].
    """
    allowed_height_range = LLCP2022_HEIGHT_RANGE

    if gender == "Male":
        return [
            max(height - 30, allowed_height_range[0]),
            min(height + 10, allowed_height_range[1]),
        ]
    else:  # Female
        return [
            max(height + 5, allowed_height_range[0]),
            min(height + 50, allowed_height_range[1]),
        ]


def get_bmi_preference(gender, bmi):
    """Deterministically obtains a weight preference.

    Args:
        gender (str): Agent gender.
        bmi (int): Agent BMI group ID.

    Returns:
        list: BMI preference range, as [min, max].
    """
    if gender == "Male":
        return [1, min(4, bmi)]
    else:  # Female
        if bmi > 1:
            return [max(2, bmi - 1), min(4, bmi + 1)]
        else:
            return [bmi, bmi + 1]
