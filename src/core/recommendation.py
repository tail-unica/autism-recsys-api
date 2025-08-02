import logging
from collections.abc import Iterable

import numpy as np
import torch
from hopwise.model.logits_processor import ConstrainedLogitsProcessorWordLevel
from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.utils import KnowledgeEvaluationType, PathLanguageModelingTokenType
from transformers import StoppingCriteria

logger = logging.getLogger("PHaSE API")


class RestrictionNotApplicable(RuntimeError):
    """
    Exception raised when restrictions cannot be applied. Mainly due to all KG elements being masked.
    """

    def __init__(self, message="Restrictions cannot be applied to the current input."):
        super().__init__(message)


class ZeroShotConstrainedLogitsProcessor(ConstrainedLogitsProcessorWordLevel):
    """
    Logits processor for zero-shot constrained generation.
    This processor applies constraints to the logits based on the provided preferences and restrictions.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenized_ckg,
        tokenized_used_ids,
        max_sequence_length,
        tokenizer,
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        self.remove_user_tokens_from_sequences = kwargs.pop("remove_user_tokens_from_sequences", False)
        self.tokenized_uids = set(
            [
                vocab[1]
                for vocab in tokenizer.get_vocab().items()
                if vocab[0].startswith(PathLanguageModelingTokenType.USER.token)
            ]
        )
        self.tokenized_ui_relation = set([kwargs.pop("tokenized_ui_relation", 1)])
        self.previous_recommendations = None
        super().__init__(
            tokenized_ckg,
            tokenized_used_ids,
            max_sequence_length,
            tokenizer,
            mask_cache_size=mask_cache_size,
            pos_candidates_cache_size=pos_candidates_cache_size,
            task=task,
            **kwargs,
        )

    def __call__(self, input_ids, scores):
        """
        Process the logits to apply constraints based on user preferences and restrictions.
        """
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        unique_input_ids = input_ids
        last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
        _, input_ids_indices, input_ids_inv = np.unique(
            input_ids.cpu().numpy()[:, -last_n_tokens:], axis=0, return_index=True, return_inverse=True
        )
        unique_input_ids = input_ids[input_ids_indices]

        full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
        for idx in range(unique_input_ids.shape[0]):
            if current_len > 2 and (  # noqa: PLR2004
                self.tokenizer.decode(unique_input_ids[idx, -1]).startswith(PathLanguageModelingTokenType.ITEM.token)
                or unique_input_ids[idx, -1] == self.tokenizer.pad_token_id
            ):
                # If the last token is an item or pad token, we ban all tokens except the pad token
                banned_mask = np.ones(len(self.tokenizer), dtype=bool)
            else:
                key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
                banned_mask = self.get_banned_mask(key, candidate_tokens)

            if banned_mask.all():
                banned_mask[self.tokenizer.pad_token_id] = False

            full_mask[idx] = banned_mask

        if current_len < self.max_sequence_length - 1 - has_bos_token:
            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            scores[full_mask] = -torch.inf

        return scores

    def process_scores_rec(self, input_ids, idx):
        """Process each score based on input length and update mask list."""
        key = self.get_current_key(input_ids, idx)
        candidate_tokens = self.get_candidates_rec(*key)
        if self.previous_recommendations is not None:
            candidate_tokens = candidate_tokens - set(self.previous_recommendations)

        if self.remove_user_tokens_from_sequences:
            # In zero-shot, we do not want explanations based on user IDs, so we remove them
            candidate_tokens = candidate_tokens - self.tokenized_uids - self.tokenized_ui_relation

        return key, list(candidate_tokens)

    def get_candidates_rec(self, key1, key2=None):
        """
        :param key1:
        :param key2: if key2 is not None, it returns entity candidates, otherwise relation candidates
        """
        if key1 in self.tokenized_ckg:
            if key2 is not None and key2 in self.tokenized_ckg[key1]:
                # return tail given head + relation
                return self.tokenized_ckg[key1][key2]
            else:
                # return relations given head
                return set(self.tokenized_ckg[key1].keys())
        else:
            # If key1 is not in tokenized_ckg, return all keys as candidates. Bad sequence will be filtered out later.
            return set(self.tokenized_ckg.keys())
            # raise ValueError(
            #     f"Key {key1} ('{self.tokenizer.convert_ids_to_tokens(key1)}') not found in tokenized_ckg"
            # )


class RestrictionLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    """
    Logits processor for applying restrictions to the logits based on user preferences and restrictions.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenized_ckg,
        tokenizer,
        propagate_connected_entities=True,
        mask_cache_size=3 * 10**4,
        pos_candidates_cache_size=1 * 10**5,
        task=KnowledgeEvaluationType.REC,
        **kwargs,
    ):
        super().__init__(
            tokenized_ckg,
            None,
            None,
            tokenizer,
            mask_cache_size=mask_cache_size,
            pos_candidates_cache_size=pos_candidates_cache_size,
            task=task,
            **kwargs,
        )
        self.propagate_connected_entities = propagate_connected_entities
        self.tokenized_entities = [
            vocab[1]
            for vocab in tokenizer.get_vocab().items()
            if vocab[0].startswith(PathLanguageModelingTokenType.ENTITY.token)
            or vocab[0].startswith(PathLanguageModelingTokenType.ITEM.token)
        ]

        self.current_restricted_candidates = []
        self.current_hard_restrictions = []
        self.current_soft_restrictions = []

    def set_restrictions(self, restricted_candidates=None, hard_restrictions=None, soft_restrictions=None):
        """
        Set the hard and soft restrictions for the logits processor.
        """
        if not restricted_candidates and not hard_restrictions and not soft_restrictions:
            raise ValueError("At least one restriction must be provided.")

        self.current_restricted_candidates = restricted_candidates or []
        self.current_hard_restrictions = hard_restrictions or []
        self.current_soft_restrictions = soft_restrictions or []

    def clear_restrictions(self):
        """
        Clear the current restrictions.
        """
        self.current_restricted_candidates = []
        self.current_hard_restrictions = []
        self.current_soft_restrictions = []

    def __call__(self, input_ids, scores):
        # restrictions are independent of input ids, so the full_mask will be repeated for each input row
        full_mask = self.gen_keepmask_restricted_candidates()

        if np.all(full_mask):
            raise RestrictionNotApplicable()
        # TODO: add check if restricted_candidates form a connected component in the CKG

        for h_rest in self.current_hard_restrictions:
            full_mask = np.logical_or(full_mask, self.gen_banmask_from_key(h_rest))

        if np.all(full_mask):
            raise RestrictionNotApplicable()

        self.current_soft_restrictions = sorted(
            self.current_soft_restrictions,
            key=lambda k: len(self.tokenized_ckg.get(k, [])),
            reverse=True,
        )

        for s_rest in self.current_soft_restrictions:
            soft_mask = np.logical_or(full_mask, self.gen_banmask_from_key(s_rest))

            if np.all(soft_mask):
                # Rollback the last soft restriction if it masks all tokens
                break
            else:
                full_mask = soft_mask

        scores[:, full_mask] = -np.inf

        return scores

    def extract_connected_entities(self, token_id):
        """
        Estrae tutte le entità connesse al token_id, sotto forma di lista completamente appiattita.
        """
        connected_entities = set()
        for entity_set in self.tokenized_ckg[token_id].values():
            connected_entities.update(entity_set if isinstance(entity_set, (list, set)) else [entity_set])

        return connected_entities

    def gen_keepmask_restricted_candidates(self):
        mask = np.zeros(len(self.tokenizer), dtype=bool)

        if self.current_restricted_candidates:
            mask[self.tokenized_entities] = True  # Ban all entities by default

            shared_connected_entities = self.extract_connected_entities(self.current_restricted_candidates[0])
            for r_candidate in self.current_restricted_candidates[1:]:
                if r_candidate in self.tokenized_ckg:
                    r_candidate_connected_entities = self.extract_connected_entities(r_candidate)
                    shared_connected_entities = shared_connected_entities & r_candidate_connected_entities

                    # for connected_entity in connected_entities:
                    #     connected_tokens = self.extract_connected_entities(connected_entity)
                    #     connected_tokens.append(connected_entity)
                    #     mask[connected_tokens] = False # connected_entities

            mask[self.current_restricted_candidates] = False  # Keep all restricted candidates
            mask[list(shared_connected_entities)] = False

        return mask

    def gen_banmask_from_key(self, token_id):
        mask = np.zeros(len(self.tokenizer), dtype=bool)

        mask[token_id] = True

        if token_id in self.tokenized_ckg and self.propagate_connected_entities:
            connected_entities = list(self.extract_connected_entities(token_id))
            mask[connected_entities] = True

            # for connected_entity in connected_entities:
            #     connected_tokens = self.extract_connected_entities(connected_entity)
            #     connected_tokens.append(connected_entity)
            #     mask[connected_tokens] = True # connected_entities

        return mask


class ZeroShotCumulativeSequenceScorePostProcessor(CumulativeSequenceScorePostProcessor):
    """
    Post-processor for zero-shot cumulative sequence scoring.
    This processor applies a cumulative score to sequences based on their relevance and diversity.
    """

    def __init__(self, tokenizer, item_num, topk=10):
        self.tokenizer = tokenizer
        self.item_num = item_num
        self.topk = topk

    def get_sequences(self, generation_outputs, user_num=1, max_new_tokens=24, previous_recommendations=None):
        normalized_scores = self.normalize_tuple(generation_outputs["scores"])
        normalized_sequences_scores = self.calculate_sequence_scores(
            normalized_scores, generation_outputs["sequences"], max_new_tokens=max_new_tokens
        )

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        valid_sequences_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))  # false if finite
        normalized_sequences_scores = torch.where(valid_sequences_mask, -torch.inf, normalized_sequences_scores)

        sorted_indices = normalized_sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_sequences_scores = normalized_sequences_scores[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]

        return self.parse_sequences(
            sorted_batch_user_index,
            sorted_sequences,
            sorted_sequences_scores,
            previous_recommendations=previous_recommendations,
        )

    def parse_sequences(self, user_index, sequences, sequences_scores, previous_recommendations=None):
        """
        Parses the sequences and their scores into a structured format.
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        for batch_uidx, sequence, sequence_score in zip(user_index, sequences, sequences_scores):
            parsed_seq = self._parse_single_sequence(
                scores, batch_uidx, sequence, previous_recommendations=previous_recommendations
            )
            if parsed_seq is None:
                continue
            recommended_item, decoded_seq = parsed_seq

            scores[batch_uidx, recommended_item] = sequence_score
            user_topk_sequences.append([batch_uidx, recommended_item, sequence_score.item(), decoded_seq])

        return scores, user_topk_sequences

    def _parse_single_sequence(self, scores, batch_uidx, sequence, previous_recommendations=None):
        """Parses a single sequence to extract user ID, recommended item, and the decoded sequence."""
        previous_recommendations = previous_recommendations or []

        seq = self.tokenizer.decode(sequence).split(" ")
        seq = list(filter(lambda x: x != self.tokenizer.pad_token, seq))

        # Bug behavior: check for consecutive duplicate tokens
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            return

        # TODO: if recommending ingredients will be a design choice to be made in the project
        # I could edit the logit processor to stop also as soon as the entity is contained in phaseapi_recipes.parquet
        # and the same thing also here if the recommended token is an entity E
        recommended_token = seq[-1]
        if (
            not recommended_token.startswith(PathLanguageModelingTokenType.ITEM.token)
            or recommended_token == self.tokenizer.pad_token
        ):
            return

        recommended_item = int(recommended_token[1:])

        if torch.isfinite(scores[batch_uidx, recommended_item]) or recommended_item in previous_recommendations:
            return

        return recommended_item, seq


class ZeroShotCriteria(StoppingCriteria):
    """
    Stopping criteria for zero-shot generation.
    This criteria stops the generation when an item
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        reached_item = [
            token.startswith(PathLanguageModelingTokenType.ITEM.token)
            for token in self.tokenizer.decode(input_ids[:, -1]).split()
        ]
        return torch.tensor(reached_item, device=input_ids.device)


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


def match_elements(
    elements, kg_elements_semantic_matcher, entity_mapping, dataset_for_tokenization=None, tag_offset=0.1
):
    """
    Match elements to the most similar items in the knowledge graph using semantic matching.
    Performs a double match: first for the element itself, then for the tag prefixed with "tag.",
    due to the semantich matcher often ignoring tags.
    """
    matched_elements = []

    if elements is None:
        return None

    elements = elements if isinstance(elements, Iterable) else [elements]

    if elements:
        for el in elements:
            match = kg_elements_semantic_matcher.find_most_similar_item(el, max_distance=1.0)
            tag_match = kg_elements_semantic_matcher.find_most_similar_item("tag." + el, max_distance=1.0)
            if match is None and tag_match is None:
                logger.warning(f"No match found for KG element: {el}")
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

                mapped_match = entity_mapping.get(match, match)
                logger.info(f"Matched '{el}' to '{match}' with mapped value '{mapped_match}'")

                if dataset_for_tokenization is not None and isinstance(mapped_match, int):
                    mapped_match = id2tokenizer_token(dataset_for_tokenization, mapped_match)

                matched_elements.append(mapped_match)
        matched_elements = list(set(matched_elements))  # Remove duplicates

    return matched_elements


def id2tokenizer_token(dataset, _id):
    if _id < dataset.item_num:
        token = PathLanguageModelingTokenType.ITEM.token + str(_id)
    else:
        token = PathLanguageModelingTokenType.ENTITY.token + str(_id)

    return dataset.tokenizer.convert_tokens_to_ids(token)


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


def prepare_recommender_and_raw_inputs(  # noqa: PLR0913
    user_id,
    dataset,
    recommender,
    kg_elements_semantic_matcher,
    existing_user_cumulative_sequence_postprocessor,
    constrained_logits_processors_list,
    zero_shot_constrained_logits_processors_list,
    zero_shot_sequence_postprocessor,
    preferences=None,
    previous_recommendations=None,
    hard_restrictions=None,
    soft_restrictions=None,
    restrict_preference_graph=False,
    tag_offset=0.1,
):
    if user_id in dataset.field2id_token[dataset.uid_field]:
        logger.info(f"User {user_id} exists in the dataset, using existing user sequence postprocessor.")
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
    else:
        logger.info(f"User {user_id} does not exist in the dataset, using zero-shot sequence postprocessor.")
        logger.info("Finding best matches for preferences and preparing raw inputs...")
        if not preferences:
            logger.error("No preferences provided for zero-shot recommendation.")
            return None
        matched_preferences = match_elements(
            elements=preferences,
            kg_elements_semantic_matcher=kg_elements_semantic_matcher,
            entity_mapping=dataset.field2token_id[dataset.entity_field],
            dataset_for_tokenization=None,
            tag_offset=tag_offset,
        )
        if not matched_preferences:
            logger.error("No valid KG elements matching provided preferences for zero-shot recommendation.")
            return None
        raw_inputs = prepare_zero_shot_raw_inputs(matched_preferences, dataset)

        recommender.sequence_postprocessor = zero_shot_sequence_postprocessor
        recommender.logits_processor_list = zero_shot_constrained_logits_processors_list

        if previous_recommendations:
            matched_previous_recommendations = match_elements(
                elements=previous_recommendations,
                kg_elements_semantic_matcher=kg_elements_semantic_matcher,
                entity_mapping=dataset.field2token_id[dataset.entity_field],
                dataset_for_tokenization=dataset,
                tag_offset=tag_offset,
            )

            for logit_processor in recommender.logits_processor_list:
                if isinstance(logit_processor, ZeroShotConstrainedLogitsProcessor):
                    logit_processor.previous_recommendations = matched_previous_recommendations

        for logit_processor in recommender.logits_processor_list:
            if isinstance(logit_processor, RestrictionLogitsProcessorWordLevel):
                if hard_restrictions or soft_restrictions:
                    logger.info("Setting restrictions")
                    matched_hard_restrictions = match_elements(
                        elements=hard_restrictions,
                        kg_elements_semantic_matcher=kg_elements_semantic_matcher,
                        entity_mapping=dataset.field2token_id["entity_id"],
                        dataset_for_tokenization=dataset,
                        tag_offset=tag_offset,
                    )

                    matched_soft_restrictions = match_elements(
                        elements=soft_restrictions,
                        kg_elements_semantic_matcher=kg_elements_semantic_matcher,
                        entity_mapping=dataset.field2token_id["entity_id"],
                        dataset_for_tokenization=dataset,
                        tag_offset=tag_offset,
                    )

                    restrictions = {}
                    if matched_hard_restrictions or matched_soft_restrictions:
                        restrictions.update(
                            dict(
                                hard_restrictions=matched_hard_restrictions,
                                soft_restrictions=matched_soft_restrictions,
                            )
                        )

                    if restrict_preference_graph:
                        tokenized_matched_preferences = [
                            id2tokenizer_token(dataset, pref) for pref in matched_preferences
                        ]
                        restrictions["restricted_candidates"] = tokenized_matched_preferences

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
