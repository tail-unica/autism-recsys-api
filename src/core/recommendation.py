import warnings

import numpy as np
import torch
from hopwise.model.logits_processor import ConstrainedLogitsProcessorWordLevel
from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.utils import KnowledgeEvaluationType, PathLanguageModelingTokenType
from transformers import StoppingCriteria


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


class RestrictionLogitsProcessorWordLevel(ConstrainedLogitsProcessorWordLevel):
    """
    Logits processor for applying restrictions to the logits based on user preferences and restrictions.
    """

    def __init__(  # noqa: PLR0913
        self,
        tokenized_ckg,
        tokenizer,
        entity_mapping,
        item_num,
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
        self.entity_mapping = entity_mapping
        self.item_num = item_num
        self.propagate_connected_entities = propagate_connected_entities

        self.current_hard_restrictions = None
        self.current_soft_restrictions = None

    def set_restrictions(self, hard_restrictions=None, soft_restrictions=None):
        """
        Set the hard and soft restrictions for the logits processor.
        """
        if not hard_restrictions and not soft_restrictions:
            raise ValueError("At least one restriction must be provided.")

        self.current_hard_restrictions = hard_restrictions
        self.current_soft_restrictions = soft_restrictions

    def clear_restrictions(self):
        """
        Clear the current restrictions.
        """
        self.current_hard_restrictions = None
        self.current_soft_restrictions = None

    def __call__(self, input_ids, scores):
        # restrictions are independent of input ids, so the full_mask will be repeated for each input row
        if self.current_hard_restrictions is None or self.current_soft_restrictions is None:
            warnings.warn(
                "No restrictions set. Returning scores without applying any restrictions.",
                UserWarning,
            )
            return scores

        full_mask = np.zeros(len(self.tokenizer), dtype=bool)

        for h_rest in self.current_hard_restrictions:
            full_mask = np.logical_or(full_mask, self.gen_banmask_from_key(h_rest))

        if np.all(full_mask):
            raise RestrictionNotApplicable()

        breakpoint()
        self.current_soft_restrictions = sorted(
            self.current_soft_restrictions,
            key=lambda k: len(self.tokenized_ckg.get(self.entity_mapping[k], [])),
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
        connected_entities = list(
            set(connected_entities)
        )  # .sorted()  # l'ordine è temporaneo, serve solo a render più leggibile lo schifo

        return connected_entities

    def gen_banmask_from_key(self, key):
        mask = np.zeros(len(self.tokenizer), dtype=bool)

        id_interno = self.entity_mapping[key]
        if id_interno < self.item_num:
            token = PathLanguageModelingTokenType.ITEM.value + str(id_interno)
        else:
            token = PathLanguageModelingTokenType.ENTITY.value + str(id_interno)

        token_id = self.tokenizer.convert_tokens_to_ids(token)
        mask[token_id] = True

        if token_id in self.tokenized_ckg and self.propagate_connected_entities:
            connected_entities = self.extract_connected_entities(token_id)
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


def prepare_zero_shot_raw_inputs(preferences, dataset, kg_elements_semantic_matcher):
    matched_preferences = []
    preferences = preferences or []
    for pref in preferences:
        matched_KG_element = kg_elements_semantic_matcher.find_most_similar_item(query=pref)
        match, _ = matched_KG_element
        matched_preferences.append(match)

    raw_inputs = [
        dataset.path_token_separator.join(
            [
                dataset.tokenizer.bos_token,
                (
                    PathLanguageModelingTokenType.ITEM.token
                    if pref in dataset.entity2item
                    else PathLanguageModelingTokenType.ENTITY.token
                )
                + str(dataset.field2token_id[dataset.entity_field][pref]),
            ]
        )
        for pref in matched_preferences
    ]

    return raw_inputs


def set_restrictions(
    hard_restrictions,
    soft_restrictions,
    kg_elements_semantic_matcher,
    zero_shot_constrained_logits_processors_list,
    logger,
):
    matched_hard_restrictions = []
    matched_soft_restrictions = []

    if hard_restrictions:
        for restriction in hard_restrictions:
            matched_element = kg_elements_semantic_matcher.find_most_similar_item(restriction, max_distance=1.0)
            if matched_element is None:
                logger.warning(f"No match found for hard restriction: {restriction}")
            else:
                match, _ = matched_element
                matched_hard_restrictions.append(match)
        matched_hard_restrictions = list(set(matched_hard_restrictions))  # Remove duplicates

    if soft_restrictions:
        for restriction in soft_restrictions:
            matched_element = kg_elements_semantic_matcher.find_most_similar_item(restriction, max_distance=1.0)
            if matched_element is None:
                logger.warning(f"No match found for soft restriction: {restriction}")
            else:
                match, _ = matched_element
                matched_soft_restrictions.append(match)
        matched_soft_restrictions = list(set(matched_soft_restrictions))  # Remove duplicates

    zero_shot_constrained_logits_processors_list[-1].set_restrictions(
        hard_restrictions=matched_hard_restrictions,
        soft_restrictions=matched_soft_restrictions,
    )


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
