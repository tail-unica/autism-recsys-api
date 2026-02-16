from collections.abc import Iterable
from numpy import arange

from hopwise.utils import PathLanguageModelingTokenType

# These imports are only needed for type checking (e.g. isinstance(logit_processor, RestrictionLogitsProcessorWordLevel))
from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
    id2tokenizer_token,
)
from src.core.logger import logger
from src.core.data import get_item_names


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

import numpy as np

def user_sample_compatible_features(aversions: dict) -> list[str]:
    """
    Generates a list of sensory features that are considered compatible based on the user's aversion levels, 
    and samples one value for each compatible feature using a strategy that favors values near the middle of the compatible range.    
    
    :param aversions: Dictionary mapping idiosyncratic aversions to the user's aversion levels (values from 1 to 5).
    :type aversions: dict[str, float]
    :return: A list of sensory features in the format "SensoryFeature.{feature}.{value}" that are compatible.
    :rtype: list[str]
    """
    LIKERT_STEP = 0.1
    # Create range [1.0, 5.0] inclusive
    LIKERT_RANGE = np.arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)

    compatible_features = {}
    
    # Identify all compatible values
    for val in LIKERT_RANGE:
        val = round(val, 1)
        # Check compatibility for all features at this value
        context = {feature: val for feature in sensory_features_map}
        compatibility = user_feature_compatibility(aversions, context)
        
        for feature, is_compatible in compatibility.items():
            if is_compatible:  # Keep only compatible ones
                compatible_features.setdefault(feature, []).append(val)

    sampled_compatible_features = []
    
    for feature, val_list in compatible_features.items():
        if not val_list:
            continue
            
        vals = np.array(val_list)
        
        # --- Circular Logic ---
        # Calculate gaps between consecutive values
        diffs = np.diff(vals)
        # Calculate the gap wrapping around the 5.0 -> 1.0 boundary
        # Gap is the empty space at the start (val[0]-1) + empty space at end (5-val[-1])
        wrap_gap = (vals[0] - 1.0) + (5.0 - vals[-1])
        
        all_gaps = np.append(diffs, wrap_gap)
        max_gap_idx = np.argmax(all_gaps)
        
        # If the largest gap is NOT the wrap-around, we must roll the array
        # so the largest gap becomes the new start/end boundary.
        if max_gap_idx != len(all_gaps) - 1:
            vals = np.roll(vals, -(max_gap_idx + 1))

        # --- Random Near Middle ---
        n = len(vals)
        mid_idx = n // 2
        
        # Apply random jitter (approx +/- 10% of array length)
        jitter_range = max(1, int(n * 0.1))
        random_offset = np.random.randint(-jitter_range, jitter_range + 1)
        
        # Select index and clip to bounds
        selected_idx = np.clip(mid_idx + random_offset, 0, n - 1)
        sampled_value = vals[selected_idx]
        
        sampled_compatible_features.append(f"SensoryFeature.{feature}.{sampled_value:.1f}")

    return sampled_compatible_features

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
    # Resolve underlying model if wrapped by torch.compile (OptimizedModule)
    model = getattr(recommender, '_orig_mod', recommender)
    model.sequence_postprocessor = existing_user_cumulative_sequence_postprocessor
    model.logits_processor_list = constrained_logits_processors_list

    return raw_inputs

def prepare_recommender_and_raw_inputs_zero_shot(  # noqa: PLR0913
    recommender,
    dataset,
    zero_shot_sequence_postprocessor,
    zero_shot_constrained_logits_processors_list,
    preferences=None,
    previous_recommendations=[],
    aversions=None,
):
    """
    Docstring per prepare_recommender_and_raw_inputs_zero_shot
    
    :param recommender: Descrizione
    :param dataset: Descrizione
    :param zero_shot_sequence_postprocessor: Descrizione
    :param zero_shot_constrained_logits_processors_list: Descrizione
    :param preferences: List of items encoded as dataset ids
    :param previous_recommendations: Descrizione
    :param aversions: Descrizione
    """
    if not preferences:
        logger.error("No preferences provided for zero-shot recommendation.")
        return None

    logger.debug("preferences: " + str(preferences))

    # Map preferences from dataset IDs to hopwise tokens
    token_iid_list = dataset.field2id_token[dataset.iid_field]
    token_to_iid = {tok: idx for idx, tok in enumerate(token_iid_list)}

    preference_ids = []
    for pref in preferences:
        #if isinstance(pref, int):
        #    preference_ids.append(pref)
        #    continue

        pref_id = token_to_iid.get(pref)
        if pref_id is None:
            logger.error(
                f"Value {pref} not found in dataset.field2id_token[{dataset.iid_field}]."
            )
            return None
        preference_ids.append(pref_id)

    # Convert preference IDs to tokenizer tokens (e.g., "55" -> "I1 -> 101")
    # Create a sequence for each preferred item to be used as input for the recommendation model
    raw_inputs = [
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                PathLanguageModelingTokenType.ITEM.token + str(pref_id)
            ]
        )
        for pref_id in preference_ids
    ]

    logger.debug(f"Raw inputs after adding preferences: {raw_inputs}")

    token_eid_list = dataset.field2id_token[dataset.entity_field]
    token_to_eid = {tok: idx for idx, tok in enumerate(token_eid_list)}

    # Add compatible sensory features to the raw inputs based on the user's aversions
    # Create a sequence for each compatible sensory feature to be used as input for the recommendation model
    raw_inputs.extend([
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                PathLanguageModelingTokenType.ENTITY.token + str(token_to_eid[feature])
            ]
        )
        for feature in user_sample_compatible_features(aversions) if feature in token_to_eid
    ])

    logger.debug(f"Raw inputs after adding preferences and compatible features: {raw_inputs}")

    # Resolve underlying model if wrapped by torch.compile (OptimizedModule)
    model = getattr(recommender, '_orig_mod', recommender)
    model.sequence_postprocessor = zero_shot_sequence_postprocessor
    model.logits_processor_list = zero_shot_constrained_logits_processors_list

    previous_recommendations.extend(preferences)  # Add preferences to previous recommendations to avoid recommending them again
    if previous_recommendations:
        previous_recommendations = id2tokenizer_token(dataset, [token_to_iid.get(item) for item in previous_recommendations], type="place")

        for logit_processor in model.logits_processor_list:
            if isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
                logit_processor.previous_recommendations = previous_recommendations


    hard_restrictions = user_feature_mask(aversions) if aversions else None
    tokenized_hard_restrictions = id2tokenizer_token(dataset, hard_restrictions, type="entity") if hard_restrictions else None

    for logit_processor in model.logits_processor_list:
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


def unpack_recommendation_sequences_tuples(sequences, dataset, user_id, better_explanations=False, **kwargs):
    recommendation_ids = [seq[1] for seq in sequences]
    scores = [seq[2] for seq in sequences]
    explanations = [seq[3] for seq in sequences] 
    logger.debug(f"{'Unpacked recommendation IDs'.rjust(27)}: {str(recommendation_ids)}") # 
    logger.debug(f"{'Unpacked scores'.rjust(27)}: {str(scores)}")
    logger.debug(f"{'Unpacked explanations'.rjust(27)}: {str(explanations)}")
    for idx in range(len(explanations)):
        explanations[idx] = [token2real_token(token, dataset) for token in explanations[idx][1:]]

    # EXPLANATIONS FORMATTING
    # 2. Ti suggeriamo [POI] perché ti è piaciuto [POI], che ha lo stesso livello di [SENSORY FEATURE].
    #    [BOS] -> [POI] [RELATION] [SENSORY FEATURE] [RELATION] [POI]
    # 4. Ti suggeriamo [POI] perché è piaciuto ad una persona [USER] a cui, come a te, è piaciuto [POI].
    #    [BOS] -> [POI] [RELATION] [USER] [RELATION] [POI]
    # 6. Ti suggeriamo [POI] perché ha un livello di [SENSORY FEATURE] compatibile con il tuo.
    #    [BOS] -> [SENSORY FEATURE] [RELATION] [POI]
    # 7. Ti suggeriamo [POI] perché è piaciuto ad una persona [USER] a cui, come a te, danno fastidio [SENSORY FEATURE].

    try:
        if better_explanations:
            force_paths = kwargs.get("force_paths", [])
            force_path_explanations = kwargs.get("force_path_explanations", [])

        if user_id not in dataset.field2id_token[dataset.uid_field]:
            explanations = [
                f"Siccome ti piace " + " ".join(exp) # .replace(dataset.ui_relation, "interacted_with")
                for exp in explanations
            ]

        mapped_recommendations = dataset.field2id_token["name"][dataset.item_feat[recommendation_ids]["name"]].tolist()
        recommendations = [" ".join(filter(lambda x: x != "[PAD]", x)) for x in mapped_recommendations]
    except Exception as e:
        logger.error(f"Error processing recommendations: {e}")
        return None

    return scores, recommendations, explanations
