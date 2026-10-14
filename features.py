import math
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np


EXTRA_FEATURES = [
    "median_msg_chars", "q25_msg_chars", "q75_msg_chars", "first_msg_chars",
    "last_msg_chars", "avg_msg_words", "max_msg_words", "question_msg_share",
    "exclaim_msg_share", "cyrillic_msg_share", "latin_msg_share",
    "empty_msg_share", "long_msg_share", "whitespace_ratio",
    "supplementary_unicode_ratio", "question_char_ratio", "exclaim_char_ratio",
    "char_entropy", "word_entropy", "shared_vocab_ratio",
    "avg_prev_overlap", "max_prev_overlap", "avg_prev_similarity",
    "max_prev_similarity", "exact_echo_share", "contains_echo_share",
    "avg_log_reply_len_ratio", "replied_to_question_share",
    "avg_self_adjacent_similarity", "first_turn_fraction", "last_turn_fraction",
    "is_last_speaker", "consecutive_own_turns",
]


def tokens(text: str) -> list[str]:
    """разбивает текст на слова без учёта регистра."""
    return re.findall(r"\w+", text.casefold())


def average(values) -> float:
    """считает среднее и возвращает ноль для пустого набора."""
    return float(np.mean(values)) if values else 0.0


def entropy(items) -> float:
    """измеряет разнообразие значений через энтропию."""
    if not items:
        return 0.0
    counts = Counter(items)
    size = len(items)
    return -sum(count / size * math.log2(count / size) for count in counts.values())


def extra_features(messages, index: int) -> dict[str, float]:
    """считает признаки текста, порядка ходов и реакции на собеседника."""
    own = [(i, message.text) for i, message in enumerate(messages) if message.participant_index == index]
    other = [message.text for message in messages if message.participant_index != index]
    texts = [text for _, text in own]
    lengths = [len(text) for text in texts]
    word_lists = [tokens(text) for text in texts]
    flat_words = [word for group in word_lists for word in group]
    full = " ".join(texts)
    other_full = " ".join(other)
    previous_other = None
    overlaps = []
    similarities = []
    exact = []
    contains = []
    length_ratios = []
    question_replies = []
    for message in messages:
        if message.participant_index != index:
            previous_other = message.text
            continue
        if previous_other is None:
            continue
        current = message.text.strip().casefold()
        previous = previous_other.strip().casefold()
        current_words = set(tokens(current))
        previous_words = set(tokens(previous))
        overlaps.append(len(current_words & previous_words) / max(len(current_words | previous_words), 1))
        similarities.append(SequenceMatcher(None, current[:500], previous[:500]).ratio())
        exact.append(int(bool(current) and current == previous))
        contains.append(int(bool(current) and bool(previous) and (current in previous or previous in current)))
        length_ratios.append((len(current) + 1) / (len(previous) + 1))
        question_replies.append(int("?" in previous))
    self_similarities = [
        SequenceMatcher(None, texts[i - 1].casefold(), texts[i].casefold()).ratio()
        for i in range(1, len(texts))
    ]
    own_vocab = set(flat_words)
    other_vocab = set(tokens(other_full))
    size = max(len(full), 1)
    values = [
        float(np.median(lengths)), float(np.quantile(lengths, .25)),
        float(np.quantile(lengths, .75)), lengths[0], lengths[-1],
        average([len(group) for group in word_lists]), max(map(len, word_lists)),
        average(["?" in text for text in texts]),
        average(["!" in text for text in texts]),
        average([bool(re.search(r"[а-яА-Я]", text)) for text in texts]),
        average([bool(re.search(r"[a-zA-Z]", text)) for text in texts]),
        average([len(text.strip()) == 0 for text in texts]),
        average([len(text) > 80 for text in texts]),
        sum(char.isspace() for char in full) / size,
        sum(ord(char) > 0xffff for char in full) / size,
        full.count("?") / size,
        full.count("!") / size,
        entropy(list(full.casefold())), entropy(flat_words),
        len(own_vocab & other_vocab) / max(len(own_vocab | other_vocab), 1),
        average(overlaps), max(overlaps, default=0),
        average(similarities), max(similarities, default=0),
        average(exact), average(contains),
        average([math.log1p(ratio) for ratio in length_ratios]),
        average(question_replies), average(self_similarities),
        own[0][0] / len(messages), own[-1][0] / len(messages),
        int(messages[-1].participant_index == index),
        sum(messages[i].participant_index == messages[i - 1].participant_index == index for i in range(1, len(messages))),
    ]
    return dict(zip(EXTRA_FEATURES, values, strict=True))
