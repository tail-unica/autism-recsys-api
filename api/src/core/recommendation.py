from collections.abc import Iterable
from numpy import arange

from hopwise.utils import PathLanguageModelingTokenType

from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
)
from src.core.logger import logger
from src.core.data import get_item_names

def id2tokenizer_token(dataset, ids, type):
    """
    Docstring per id2tokenizer_token
    
    :param dataset: Hopwise dataset object
    :param ids: List of IDs to convert into tokens as they appear in the atomic files.
    :type ids: list[str]
    :param type: place, entity, relation, user
    """
    def place():
        # example: "55" -> "I1"
        token_iid_list = dataset.field2id_token[dataset.iid_field]
        return {tok: idx for idx, tok in enumerate(token_iid_list)}

    def entity():
        # example: "SensoryFeature.NOISE.2.3" -> "R789"
        token_eid_list = dataset.field2id_token[dataset.entity_field]
        return {tok: idx for idx, tok in enumerate(token_eid_list)}
    
    def relation():
        # example: "HAS_SENSORY_FEATURE" -> "R1"
        token_rid_list = dataset.field2id_token[dataset.relation_field]
        return {tok: idx for idx, tok in enumerate(token_rid_list)}

    def user():
        # example: "474" -> "U42"
        raise NotImplementedError("User type not implemented yet.")
    
    type_function_map = {
        "place": place,
        "entity": entity,
        "relation": relation,
        "user": user,
    }

    type_token_map = {
        "place": PathLanguageModelingTokenType.ITEM.token,
        "entity": PathLanguageModelingTokenType.ENTITY.token,
        "relation": PathLanguageModelingTokenType.RELATION.token,
        "user": PathLanguageModelingTokenType.USER.token,
    }

    if type not in type_function_map:
        raise ValueError(f"Type {type} not recognized. Available types: {list(type_function_map.keys())}")
    
    token_map = type_function_map[type]()
    token_prefix = type_token_map[type]
    
    # Convert ids to tokenizer tokens (e.g., "55" -> "I1 -> 101")
    return [dataset.tokenizer.convert_tokens_to_ids(token_prefix + str(token_map[id])) for id in ids if id in token_map]


def token2real_token(token, dataset):
    if token.startswith(PathLanguageModelingTokenType.ITEM.token):
        item_name = dataset.field2id_token["name"][dataset.item_feat[int(token[1:])]["name"]]
        item_name = " ".join(filter(lambda x: x != "[PAD]", item_name))
        token = item_name
    elif token.startswith(PathLanguageModelingTokenType.ENTITY.token):
        token = dataset.field2id_token[dataset.entity_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.RELATION.token):
        token = dataset.field2id_token[dataset.relation_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.USER.token):
        token = dataset.field2id_token[dataset.uid_field][int(token[1:])]

    return token

sensory_features_map: dict[str, list[str]] = {
    "LIGHT": ["bright_light", "dim_light"],
    "SPACE": ["wide_space", "narrow_space"],
    "CROWD": ["crowd"],
    "NOISE": ["noise"],
    "ODOR": ["odor"],
}

def user_feature_compatibility(aversions: dict[str, float], features: dict[str, float]) -> dict[str, bool]:
    """
    Calculates the compatibility level between the user's sensory aversion values 
    and the sensory features used in the knowledge graph.

    :param aversions: Dictionary mapping idiosyncratic aversions to the user's aversion levels (values from 1 to 5).
    :type aversions: dict[str, int]
    :param features: Dictionary representing the sensory features with their respective values.
    :type features: dict[str, float]
    :returns: Dictionary associating each sensory feature with a boolean indicating compatibility.
    :rtype: dict[str, bool]
    """

    INDIVIDUAL_COMPATIBILITY_THRESHOLD = 3
    
    def compute_aversion_high(ft_value, ua):
        return 1 + (ua - 1) * (ft_value - 1) / (5 - 1)

    def compute_aversion_low(ft_value, ua):
        return 1 + (ft_value - 5) * (1 - ua) / (5 - 1)
    
    def compute_aversion_low_high(low_av, high_av):
        return max(low_av, high_av)

    sensory_features_compatibility = {}

    for feature, aversions_list in sensory_features_map.items():
        # f^up strategy
        if len(aversions_list) == 2:
            low_aversion = aversions.get(aversions_list[0], 1.0)
            high_aversion = aversions.get(aversions_list[1], 1.0)
            sensory_features_compatibility[feature] = compute_aversion_low_high(
                compute_aversion_low(features[feature], low_aversion),
                compute_aversion_high(features[feature], high_aversion),
            ) > INDIVIDUAL_COMPATIBILITY_THRESHOLD
        # f^V strategy
        elif len(aversions_list) == 1:
            aversion = aversions.get(aversions_list[0], 1.0)
            sensory_features_compatibility[feature] = compute_aversion_high(features[feature], aversion) > INDIVIDUAL_COMPATIBILITY_THRESHOLD
    
    return sensory_features_compatibility

def user_feature_mask(aversions: dict[str, float]) -> list[str]:
    """
    Generates a list of sensory features that are considered non-compatible based on the user's aversion levels.

    :returns: A list of sensory features in the format "SensoryFeature.{feature}.{value}" that are non-compatible.
    :rtype: list[str]
    """
    LIKERT_STEP = .1
    LIKERT_RANGE = arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)

    non_compatible_features = set()

    for feature_value in LIKERT_RANGE:
        compatibility = user_feature_compatibility(aversions, {feature: feature_value for feature in sensory_features_map})
        for feature, is_compatible in compatibility.items():
            if not is_compatible:
                non_compatible_features.add(f"SensoryFeature.{feature}.{feature_value:.1f}")

    return list(non_compatible_features) # example: ["SensoryFeature.NOISE.2.3", "SensoryFeature.LIGHT.4.0", ...]


def user_poi_compatibility(aversions: dict, strategy: "min"):
    pass


def prepare_recommender_and_raw_inputs_existing_user(
    recommender,
    dataset,
    existing_user_cumulative_sequence_postprocessor,
    constrained_logits_processors_list,
    user_id,
):  
    logger.info("Preparing raw inputs for existing user...")
    ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
    raw_inputs = [
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                PathLanguageModelingTokenType.USER.token + user_id,
                PathLanguageModelingTokenType.RELATION.token + str(ui_relation),
            ]
        )
    ]
    recommender.sequence_postprocessor = existing_user_cumulative_sequence_postprocessor
    recommender.logits_processor_list = constrained_logits_processors_list

    return raw_inputs

def prepare_recommender_and_raw_inputs_zero_shot(  # noqa: PLR0913
    recommender,
    dataset,
    zero_shot_sequence_postprocessor,
    zero_shot_constrained_logits_processors_list,
    preferences=None,
    previous_recommendations=None,
    aversions=None,
):
    if not preferences:
        logger.error("No preferences provided for zero-shot recommendation.")
        return None

    token_iid_list = dataset.field2id_token[dataset.iid_field]
    token_to_iid = {tok: idx for idx, tok in enumerate(token_iid_list)}

    preference_ids = []
    for pref in preferences:
        if isinstance(pref, int):
            preference_ids.append(pref)
            continue

        pref_id = token_to_iid.get(pref)
        if pref_id is None:
            logger.error(
                f"Value {pref} not found in dataset.field2id_token[{dataset.iid_field}]."
            )
            return None
        preference_ids.append(pref_id)

    raw_inputs = [
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                PathLanguageModelingTokenType.ITEM.token + str(pref_id)
            ]
        )
        for pref_id in preference_ids
    ]

    recommender.sequence_postprocessor = zero_shot_sequence_postprocessor
    recommender.logits_processor_list = zero_shot_constrained_logits_processors_list

    
    if previous_recommendations:
        raise NotImplementedError("Previous recommendations for zero-shot not implemented yet.")
        previous_recommendations = id2tokenizer_token(dataset, previous_recommendations)

        for logit_processor in recommender.logits_processor_list:
            if isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
                logit_processor.previous_recommendations = previous_recommendations


    hard_restrictions = user_feature_mask(aversions) if aversions else None
    logger.debug(f"Hard restrictions (real_tokens):\n{hard_restrictions}")

    tokenized_hard_restrictions = id2tokenizer_token(dataset, hard_restrictions, type="entity") if hard_restrictions else None
    logger.debug(f"Hard restrictions (tokenized):\n{tokenized_hard_restrictions}")

    for logit_processor in recommender.logits_processor_list:
        if isinstance(logit_processor, RestrictionLogitsProcessorWordLevel):
            if hard_restrictions:
                logger.info("Setting restrictions")

                restrictions = {}
                if tokenized_hard_restrictions:
                    restrictions.update(
                        dict(
                            hard_restrictions=tokenized_hard_restrictions,
                        )
                    )

                logit_processor.set_restrictions(**restrictions)

    return raw_inputs


def reset_logits_processors(logits_processor_list):
    """Clear restrictions and previous recommendations in logits processors."""
    for logit_processor in logits_processor_list:
        if isinstance(logit_processor, RestrictionLogitsProcessorWordLevel):
            # Clear restrictions after generation
            logit_processor.clear_restrictions()
        elif isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
            # Clear previous recommendations after generation
            logit_processor.previous_recommendations = None


def unpack_recommendation_sequences_tuples(sequences, dataset, user_id):
    recommendation_ids = [seq[1] for seq in sequences]
    scores = [seq[2] for seq in sequences]
    explanations = [seq[3] for seq in sequences]
    logger.debug(f"{'Unpacked recommendation IDs'.rjust(27)}: {str(recommendation_ids)}") # 
    logger.debug(f"{'Unpacked scores'.rjust(27)}: {str(scores)}")
    logger.debug(f"{'Unpacked explanations'.rjust(27)}: {str(explanations)}")
    for idx in range(len(explanations)):
        explanations[idx] = [token2real_token(token, dataset) for token in explanations[idx][1:]]

    try:
        if user_id not in dataset.field2id_token[dataset.uid_field]:
            explanations = [
                f"User {user_id} has_preference " + " ".join(exp) # .replace(dataset.ui_relation, "interacted_with")
                for exp in explanations
            ]

        mapped_recommendations = dataset.field2id_token["name"][dataset.item_feat[recommendation_ids]["name"]].tolist()
        recommendations = [" ".join(filter(lambda x: x != "[PAD]", x)) for x in mapped_recommendations]
    except Exception as e:
        logger.error(f"Error processing recommendations: {e}")
        return None

    return scores, recommendations, explanations
