from collections.abc import Iterable

from hopwise.utils import PathLanguageModelingTokenType

from src.core import (
    cfg,
    constrained_logits_processors_list,
    existing_user_cumulative_sequence_postprocessor,
    kg_elements_semantic_matcher,
    no_id_kg_elements_map,
    recommender,
    zero_shot_constrained_logits_processors_list,
    zero_shot_sequence_postprocessor,
    # zero_shot_stop_criteria_list,
)
from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
)
from src.core.utils import logger


def prepare_zero_shot_raw_inputs(matched_preferences, dataset):
    raw_inputs = [
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                (
                    PathLanguageModelingTokenType.ITEM.token
                    if dataset.field2id_token[dataset.entity_field][pref] in dataset.entity2item
                    else PathLanguageModelingTokenType.ENTITY.token
                )
                + str(pref),
            ]
        )
        for pref in matched_preferences
    ]

    return raw_inputs


def match_elements(dataset, elements, matcher, check_tag=True, tag_offset=0.1):
    """
    Match elements to the most similar items in the knowledge graph using semantic matching.
    Performs a double match: first for the element itself, then for the tag prefixed with "tag.",
    due to the semantich matcher often ignoring tags.
    """
    matched_elements = []
    entity_mapping = dataset.field2token_id[dataset.entity_field]

    if elements is None:
        return None

    elements = elements if isinstance(elements, Iterable) else [elements]
    if elements:
        for el in elements:
            match = matcher.find_most_similar_item(el, max_distance=1.0)
            if check_tag:
                tag_match = matcher.find_most_similar_item("tag." + el, max_distance=1.0)
            else:
                tag_match = None
            if match is None and tag_match is None:
                logger.warning(f"No match found for KG element: {el}")
                continue

            if match in entity_mapping:
                mapped_match = [entity_mapping[match]]
            elif tag_match in entity_mapping:
                mapped_match = [entity_mapping[tag_match]]
            else:
                if match is None:
                    match = tag_match
                elif tag_match is not None:
                    match_name, match_dist = match
                    _, tag_match_dist = tag_match
                    # Indicator priority
                    if tag_match_dist + tag_offset < match_dist and not match_name.startswith("indicator."):
                        match = tag_match
                match, _ = match

                matches_with_id = no_id_kg_elements_map.get(match, [match])
                mapped_match = [entity_mapping.get(m, m) for m in matches_with_id]

            logger.info(f"Matched '{el}' to '{matches_with_id}' with mapped value(s) '{mapped_match}'")

            matched_elements.extend(mapped_match)
        matched_elements = list(set(matched_elements))  # Remove duplicates

    return matched_elements


def id2tokenizer_token(dataset, _ids):
    _ids = [_ids] if not isinstance(_ids, Iterable) else _ids
    tokens = []
    for _id in _ids:
        if _id < dataset.item_num:
            token = PathLanguageModelingTokenType.ITEM.token + str(_id)
        else:
            token = PathLanguageModelingTokenType.ENTITY.token + str(_id)

        token = dataset.tokenizer.convert_tokens_to_ids(token)
        tokens.append(token)
    return tokens


def token2real_token(token, dataset):
    if token.startswith(PathLanguageModelingTokenType.ITEM.token):
        item_name = dataset.field2id_token["name"][dataset.item_feat[int(token[1:])]["name"]]
        item_name = " ".join(filter(lambda x: x != "[PAD]", item_name))
        token = item_name
    elif token.startswith(PathLanguageModelingTokenType.ENTITY.token):
        token = dataset.field2id_token[dataset.entity_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.RELATION.token):
        token = dataset.field2id_token[dataset.relation_field][int(token[1:])]

    return token


def prepare_recommender_and_raw_inputs_existing_user(
    user_id,
    dataset,
    recommender,
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

    return raw_inputs, None


def prepare_recommender_and_raw_inputs_zero_shot(  # noqa: PLR0913
    dataset,
    preferences=None,
    previous_recommendations=None,
    hard_restrictions=None,
    soft_restrictions=None,
    restrict_preference_graph=False,
):
    logger.info("Finding best matches for preferences and preparing raw inputs...")
    tag_offset = cfg.recommender.tag_offset

    if not preferences:
        logger.error("No preferences provided for zero-shot recommendation.")
        return None
    matched_preferences = match_elements(
        dataset,
        elements=preferences,
        matcher=kg_elements_semantic_matcher,
        tag_offset=tag_offset,
    )
    if not matched_preferences:
        logger.error("No valid KG elements matching provided preferences for zero-shot recommendation.")
        return None
    raw_inputs = prepare_zero_shot_raw_inputs(matched_preferences, dataset)

    recommender.sequence_postprocessor = zero_shot_sequence_postprocessor
    recommender.logits_processor_list = zero_shot_constrained_logits_processors_list

    matched_previous_recommendations = []
    if previous_recommendations:
        logger.info("Finding matches for previous recommendations to remove them from candidates...")
        recipe_previous_recommendations = ["recipe." + pr for pr in previous_recommendations]
        matched_previous_recommendations = match_elements(
            dataset,
            elements=recipe_previous_recommendations,
            matcher=kg_elements_semantic_matcher,
            tag_offset=tag_offset,
        )

        tokenized_matched_previous_recommendations = id2tokenizer_token(dataset, matched_previous_recommendations)

        for logit_processor in recommender.logits_processor_list:
            if isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
                logit_processor.previous_recommendations = tokenized_matched_previous_recommendations

    for logit_processor in recommender.logits_processor_list:
        if isinstance(logit_processor, RestrictionLogitsProcessorWordLevel):
            if hard_restrictions or soft_restrictions:
                logger.info("Setting restrictions")
                matched_hard_restrictions = match_elements(
                    dataset,
                    elements=hard_restrictions,
                    matcher=kg_elements_semantic_matcher,
                    tag_offset=tag_offset,
                )
                tokenized_matched_hard_restrictions = id2tokenizer_token(dataset, matched_hard_restrictions)

                matched_soft_restrictions = match_elements(
                    dataset,
                    elements=soft_restrictions,
                    matcher=kg_elements_semantic_matcher,
                    tag_offset=tag_offset,
                )
                tokenized_matched_soft_restrictions = id2tokenizer_token(dataset, matched_soft_restrictions)

                restrictions = {}
                if tokenized_matched_hard_restrictions or tokenized_matched_soft_restrictions:
                    restrictions.update(
                        dict(
                            hard_restrictions=tokenized_matched_hard_restrictions,
                            soft_restrictions=tokenized_matched_soft_restrictions,
                        )
                    )

                if restrict_preference_graph:
                    tokenized_matched_preferences = id2tokenizer_token(dataset, matched_preferences)
                    restrictions["restricted_candidates"] = tokenized_matched_preferences

                    logit_processor.set_restrictions(**restrictions)

    return raw_inputs, matched_previous_recommendations


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
    for idx in range(len(explanations)):
        explanations[idx] = [token2real_token(token, dataset) for token in explanations[idx][1:]]

    try:
        if user_id not in dataset.field2id_token[dataset.uid_field]:
            explanations = [
                f"User {user_id} has_preference " + " ".join(exp).replace(dataset.ui_relation, "interacted_with")
                for exp in explanations
            ]

        mapped_recommendations = dataset.field2id_token["name"][dataset.item_feat[recommendation_ids]["name"]].tolist()
        recommendations = [" ".join(filter(lambda x: x != "[PAD]", x)) for x in mapped_recommendations]
    except Exception as e:
        logger.error(f"Error processing recommendations: {e}")
        return None

    return scores, recommendations, explanations
