from collections.abc import Iterable

from hopwise.utils import PathLanguageModelingTokenType

from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
)
from src.core.logger import logger


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

def prepare_recommender_and_raw_inputs_zero_shot(  # noqa: PLR0913
    recommender,
    dataset,
    zero_shot_sequence_postprocessor,
    zero_shot_constrained_logits_processors_list,
    preferences=None,
    previous_recommendations=None,
    hard_restrictions=None,
    soft_restrictions=None,
    restrict_preference_graph=False,
):

    if not preferences:
        logger.error("No preferences provided for zero-shot recommendation.")
        return None

    raw_inputs = prepare_zero_shot_raw_inputs(preferences, dataset)

    recommender.sequence_postprocessor = zero_shot_sequence_postprocessor
    recommender.logits_processor_list = zero_shot_constrained_logits_processors_list

    if previous_recommendations:
        previous_recommendations = id2tokenizer_token(dataset, previous_recommendations)

        for logit_processor in recommender.logits_processor_list:
            if isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
                logit_processor.previous_recommendations = previous_recommendations

    for logit_processor in recommender.logits_processor_list:
        if isinstance(logit_processor, RestrictionLogitsProcessorWordLevel):
            if hard_restrictions or soft_restrictions:
                logger.info("Setting restrictions")
                tokenized_hard_restrictions = id2tokenizer_token(dataset, hard_restrictions) if hard_restrictions else []
                tokenized_soft_restrictions = id2tokenizer_token(dataset, soft_restrictions) if soft_restrictions else []

                restrictions = {}
                if tokenized_hard_restrictions or tokenized_soft_restrictions:
                    restrictions.update(
                        dict(
                            hard_restrictions=tokenized_hard_restrictions,
                            soft_restrictions=tokenized_soft_restrictions,
                        )
                    )

                if restrict_preference_graph:
                    tokenized_preferences = id2tokenizer_token(dataset, preferences)
                    restrictions["restricted_candidates"] = tokenized_preferences

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
