# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import string


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    def lower(value: str) -> str:
        return value.lower()

    def replace_num_words(value: str) -> str:
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }
        pattern = re.compile(r"\b(" + "|".join(word_to_number.keys()) + r")\b")
        return pattern.sub(lambda match: word_to_number[match.group()], value)

    return replace_num_words(white_space_fix(remove_articles(remove_punctuation(lower(text)))))


def include_score(prediction: str, reference: str) -> float:
    return float(normalize_answer(reference) in normalize_answer(prediction))


def calculate_metrics(df) -> dict[str, float | int]:
    scores = [
        include_score(str(row["predicted_answer"]), str(row["answer"]))
        for _, row in df[["predicted_answer", "answer"]].iterrows()
    ]
    total = len(scores)
    accuracy = sum(scores) / total if total > 0 else 0.0
    return {"accuracy": accuracy, "total": total}

