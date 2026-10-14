import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from pydantic import BaseModel, Field
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from features import EXTRA_FEATURES, extra_features


ROOT = Path(__file__).resolve().parent
BASE_FEATURES = [
    "n_msg", "other_n", "chars", "avg_msg_chars", "max_msg_chars",
    "std_msg_chars", "unique_msg", "words", "unique_words", "punct",
    "digit", "uppercase", "cyrillic", "latin", "newline", "parentheses",
    "first_speaker", "char_diff", "n_diff",
]
FEATURES = BASE_FEATURES + EXTRA_FEATURES


class Message(BaseModel):
    text: str
    message: int
    participant_index: Literal[0, 1]


class Settings(BaseModel):
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "outputs" / "v2"
    folds: int = Field(default=4, ge=2, le=10)
    seed: int = 23
    iterations: int = Field(default=650, ge=50)
    svd_iterations: int = Field(default=600, ge=50)
    svd_components: int = Field(default=48, ge=2)
    svd_weight: float = Field(default=0.75, ge=0, le=1)
    text_weight: float = Field(default=0.05, ge=0, le=1)


def read_dialogues(path: Path) -> dict[str, list[Message]]:
    """читает диалоги и проверяет структуру реплик."""
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    result = {}
    for dialog_id, messages in raw.items():
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Empty or invalid dialogue: {dialog_id}")
        parsed = [Message.model_validate(message) for message in messages]
        if {message.participant_index for message in parsed} != {0, 1}:
            raise ValueError(f"Dialogue must contain both participants: {dialog_id}")
        result[dialog_id] = parsed
    return result


def participant_row(dialog_id: str, messages: list[Message], index: int) -> dict:
    """собирает текст и базовые признаки одного участника."""
    own = [message.text for message in messages if message.participant_index == index]
    other = [message.text for message in messages if message.participant_index != index]
    full = " \n ".join(own)
    other_full = " \n ".join(other)
    lengths = [len(text) for text in own]
    words = re.findall(r"\w+", full.lower())
    size = max(len(full), 1)
    row = {
        "dialog_id": dialog_id,
        "participant_index": index,
        "text": full,
        "n_msg": len(own),
        "other_n": len(other),
        "chars": len(full),
        "avg_msg_chars": sum(lengths) / len(lengths),
        "max_msg_chars": max(lengths),
        "std_msg_chars": float(np.std(lengths)),
        "unique_msg": len(set(own)) / len(own),
        "words": len(words),
        "unique_words": len(set(words)) / max(len(words), 1),
        "punct": sum(char in "!?.,;:" for char in full) / size,
        "digit": sum(char.isdigit() for char in full) / size,
        "uppercase": sum(char.isupper() for char in full) / size,
        "cyrillic": sum("а" <= char.lower() <= "я" for char in full) / size,
        "latin": sum("a" <= char.lower() <= "z" for char in full) / size,
        "newline": full.count("\n") / size,
        "parentheses": sum(char in "()" for char in full) / size,
        "first_speaker": int(messages[0].participant_index == index),
        "char_diff": len(full) - len(other_full),
        "n_diff": len(own) - len(other),
    }
    row.update(extra_features(messages, index))
    return row


def make_tables(dialogues: dict[str, list[Message]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """превращает диалоги в таблицы реплик и участников."""
    messages = [
        {
            "dialog_id": dialog_id,
            "message": item.message,
            "participant_index": item.participant_index,
            "text": item.text,
        }
        for dialog_id, dialogue in dialogues.items()
        for item in dialogue
    ]
    participants = [
        participant_row(dialog_id, dialogue, index)
        for dialog_id, dialogue in dialogues.items()
        for index in (0, 1)
    ]
    return pl.DataFrame(messages), pl.DataFrame(participants)


def prepare_data(settings: Settings) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """связывает диалоги с метками, проверяет и сохраняет таблицы."""
    train_dialogues = read_dialogues(settings.data_dir / "train.json")
    test_dialogues = read_dialogues(settings.data_dir / "test.json")
    train_messages, train_features = make_tables(train_dialogues)
    test_messages, test_features = make_tables(test_dialogues)

    labels = pl.read_csv(settings.data_dir / "ytrain.csv")
    test_ids = pl.read_csv(settings.data_dir / "ytest.csv")
    keys = ["dialog_id", "participant_index"]
    if labels.select(keys).unique().height != labels.height:
        raise ValueError("Duplicate training labels")
    if test_ids.select(keys).unique().height != test_ids.height:
        raise ValueError("Duplicate test IDs")
    train = train_features.join(labels, on=keys, how="left", validate="1:1")
    test = test_ids.with_row_index("_order").join(
        test_features, on=keys, how="left", validate="1:1"
    ).sort("_order").drop("_order")
    if train.height != labels.height or train.get_column("is_bot").null_count():
        raise ValueError("Training dialogues and labels do not match")
    if test.height != test_ids.height or test.get_column("text").null_count():
        raise ValueError("Test dialogues and IDs do not match")
    if set(train.get_column("is_bot").unique().to_list()) != {0, 1}:
        raise ValueError("Unexpected target labels")
    expected_ids = [f"{d}_{i}" for d, i in zip(test["dialog_id"], test["participant_index"])]
    if test.get_column("ID").to_list() != expected_ids:
        raise ValueError("Invalid test IDs")

    processed = settings.output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    train_messages.write_parquet(processed / "train_messages.parquet")
    test_messages.write_parquet(processed / "test_messages.parquet")
    train.write_parquet(processed / "train_participants.parquet")
    test.write_parquet(processed / "test_participants.parquet")

    pair_labels = [
        "".join(str(value) for value in pair)
        for pair in train.sort(keys).get_column("is_bot").to_numpy().reshape(-1, 2)
    ]
    report = {
        "train_dialogues": len(train_dialogues),
        "test_dialogues": len(test_dialogues),
        "train_participants": train.height,
        "test_participants": test.height,
        "train_messages": train_messages.height,
        "test_messages": test_messages.height,
        "bot_share": float(train.get_column("is_bot").mean()),
        "label_pairs": dict(Counter(pair_labels)),
        "empty_train_messages": int(train_messages.filter(pl.col("text").str.strip_chars() == "").height),
        "duplicate_train_texts": int(train_messages.height - train_messages.get_column("text").n_unique()),
        "median_train_message_chars": float(train_messages.get_column("text").str.len_chars().median()),
        "median_train_dialogue_messages": float(
            train_messages.group_by("dialog_id").len().get_column("len").median()
        ),
    }
    (settings.output_dir / "eda.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return train.sort(keys), test, report


def row_indices(dialogue_indices: np.ndarray) -> np.ndarray:
    """переводит индексы диалогов в индексы их участников."""
    return (2 * dialogue_indices[:, None] + np.array([0, 1])).reshape(-1)


def new_catboost(settings: Settings, fold: int, iterations: int) -> CatBoostClassifier:
    """создаёт модель бустинга с общими параметрами эксперимента."""
    return CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=4,
        loss_function="Logloss",
        random_seed=42 + fold,
        thread_count=min(4, os.cpu_count() or 1),
        verbose=False,
    )


def new_char_vectorizer() -> TfidfVectorizer:
    """создаёт векторизатор для сочетаний символов."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )


def new_word_vectorizer() -> TfidfVectorizer:
    """создаёт векторизатор для слов и пар слов."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )


def adjust_pair(probabilities: np.ndarray) -> np.ndarray:
    """убирает невозможный случай с двумя ботами в одном диалоге."""
    pair = np.clip(probabilities.reshape(-1, 2), 1e-6, 1 - 1e-6)
    denominator = 1 - pair[:, 0] * pair[:, 1]
    adjusted = np.column_stack((
        pair[:, 0] * (1 - pair[:, 1]) / denominator,
        pair[:, 1] * (1 - pair[:, 0]) / denominator,
    ))
    return adjusted.reshape(-1)


def log_experiment(name: str, score: float, settings: Settings, report: dict) -> str:
    """записывает параметры и качество варианта в журнал экспериментов."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "folds": settings.folds,
            "seed": settings.seed,
            "iterations": settings.iterations,
            "svd_iterations": settings.svd_iterations,
            "svd_components": settings.svd_components,
            "svd_weight": settings.svd_weight,
            "text_weight": settings.text_weight,
            "train_dialogues": report["train_dialogues"],
        })
        mlflow.log_metric("oof_logloss", score)
        mlflow.set_tag("candidate", name)
        return run.info.run_id


def run(settings: Settings) -> dict:
    """оценивает варианты на фолдах, обучает модели и сохраняет сабмит."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, report = prepare_data(settings)
    labels = train.get_column("is_bot").to_numpy()
    ids = train.get_column("dialog_id").to_list()
    if any(ids[2 * i] != ids[2 * i + 1] for i in range(len(ids) // 2)):
        raise ValueError("Training rows are not paired by dialogue")
    if (labels.reshape(-1, 2).sum(axis=1) > 1).any():
        raise ValueError("Training data contains a dialogue with two bots")

    numeric = train.select(FEATURES).to_numpy().astype(float)
    texts = train.get_column("text").to_list()
    pair_classes = labels.reshape(-1, 2) @ np.array([2, 1])
    cat_oof = np.zeros(len(train))
    svd_oof = np.zeros(len(train))
    text_oof = np.zeros(len(train))
    baseline_oof = np.zeros(len(train))
    folds = StratifiedKFold(settings.folds, shuffle=True, random_state=settings.seed)
    for fold, (train_dialogues, val_dialogues) in enumerate(
        folds.split(np.arange(len(pair_classes)), pair_classes)
    ):
        tr = row_indices(train_dialogues)
        va = row_indices(val_dialogues)
        baseline_oof[va] = labels[tr].mean()

        cat = new_catboost(settings, fold, settings.iterations)
        cat.fit(numeric[tr], labels[tr])
        cat_oof[va] = cat.predict_proba(numeric[va])[:, 1]

        train_texts = [texts[i] for i in tr]
        val_texts = [texts[i] for i in va]
        char_vectorizer = new_char_vectorizer()
        word_vectorizer = new_word_vectorizer()
        char_train = char_vectorizer.fit_transform(train_texts)
        char_val = char_vectorizer.transform(val_texts)
        word_train = word_vectorizer.fit_transform(train_texts)
        word_val = word_vectorizer.transform(val_texts)
        svd = TruncatedSVD(n_components=settings.svd_components, n_iter=5, random_state=42 + fold)
        svd_train = svd.fit_transform(char_train)
        svd_val = svd.transform(char_val)
        svd_cat = new_catboost(settings, fold, settings.svd_iterations)
        svd_cat.fit(np.hstack([numeric[tr], svd_train]), labels[tr])
        svd_oof[va] = svd_cat.predict_proba(np.hstack([numeric[va], svd_val]))[:, 1]

        text_model = LogisticRegression(C=10, max_iter=1000)
        text_model.fit(hstack([word_train, char_train]).tocsr(), labels[tr])
        text_oof[va] = text_model.predict_proba(hstack([word_val, char_val]).tocsr())[:, 1]
        print(f"fold {fold + 1}/{settings.folds} done", flush=True)

    structured = (1 - settings.svd_weight) * cat_oof + settings.svd_weight * svd_oof
    blended = (1 - settings.text_weight) * structured + settings.text_weight * text_oof
    candidates = {
        "cat_extended": cat_oof,
        "svd_catboost": svd_oof,
        "word_char_tfidf": text_oof,
        "blend": blended,
        "blend_pair": adjust_pair(blended),
    }
    scores = {"constant": float(log_loss(labels, baseline_oof))}
    scores.update({name: float(log_loss(labels, prob)) for name, prob in candidates.items()})
    selected = min(candidates, key=lambda name: scores[name])
    pl.DataFrame({
        "dialog_id": train.get_column("dialog_id"),
        "participant_index": train.get_column("participant_index"),
        "is_bot": labels,
        "cat_extended": cat_oof,
        "svd_catboost": svd_oof,
        "word_char_tfidf": text_oof,
        "blend_pair": candidates["blend_pair"],
    }).write_parquet(settings.output_dir / "oof_predictions.parquet")

    database = (settings.output_dir / "mlflow.db").resolve()
    artifact_dir = (settings.output_dir / "mlartifacts").resolve()
    artifact_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    if mlflow.get_experiment_by_name("bot_detection") is None:
        mlflow.create_experiment("bot_detection", artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment("bot_detection")
    run_ids = {
        name: log_experiment(name, score, settings, report)
        for name, score in scores.items()
    }

    final_cat = new_catboost(settings, settings.folds, settings.iterations)
    final_cat.fit(numeric, labels)
    cat_test = final_cat.predict_proba(test.select(FEATURES).to_numpy().astype(float))[:, 1]

    final_char_vectorizer = new_char_vectorizer()
    final_word_vectorizer = new_word_vectorizer()
    char_train = final_char_vectorizer.fit_transform(texts)
    word_train = final_word_vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=settings.svd_components, n_iter=5, random_state=42 + settings.folds)
    svd_train = svd.fit_transform(char_train)
    final_svd_cat = new_catboost(settings, settings.folds, settings.svd_iterations)
    final_svd_cat.fit(np.hstack([numeric, svd_train]), labels)

    final_text = LogisticRegression(C=10, max_iter=1000)
    final_text.fit(hstack([word_train, char_train]).tocsr(), labels)
    test_texts = test.get_column("text").to_list()
    char_test = final_char_vectorizer.transform(test_texts)
    word_test = final_word_vectorizer.transform(test_texts)
    svd_test = svd.transform(char_test)
    svd_pred = final_svd_cat.predict_proba(np.hstack([
        test.select(FEATURES).to_numpy().astype(float), svd_test
    ]))[:, 1]
    text_test = final_text.predict_proba(hstack([word_test, char_test]).tocsr())[:, 1]
    test_structured = (1 - settings.svd_weight) * cat_test + settings.svd_weight * svd_pred
    test_blend = (1 - settings.text_weight) * test_structured + settings.text_weight * text_test
    final_predictions = {
        "cat_extended": cat_test,
        "svd_catboost": svd_pred,
        "word_char_tfidf": text_test,
        "blend": test_blend,
        "blend_pair": adjust_pair(test_blend),
    }[selected]

    sample = pl.read_csv(settings.data_dir / "sample_submission.csv")
    submission = pl.DataFrame({"ID": test.get_column("ID"), "is_bot": final_predictions})
    if set(submission["ID"].to_list()) != set(sample["ID"].to_list()):
        raise ValueError("Submission IDs do not match the sample")
    if not np.isfinite(final_predictions).all() or not ((0 <= final_predictions) & (final_predictions <= 1)).all():
        raise ValueError("Invalid probabilities in submission")
    submission_path = settings.output_dir / "submission.csv"
    submission.write_csv(submission_path)

    models = settings.output_dir / "models"
    models.mkdir(exist_ok=True)
    final_cat.save_model(str(models / "catboost_extended.cbm"))
    final_svd_cat.save_model(str(models / "catboost_svd.cbm"))
    joblib.dump((final_word_vectorizer, final_char_vectorizer, svd, final_text), models / "text_models.joblib")
    results = {
        "selected_model": selected,
        "oof_logloss": scores,
        "svd_weight": settings.svd_weight,
        "text_weight": settings.text_weight,
        "mlflow_run_ids": run_ids,
        "submission": str(submission_path),
        "rows": submission.height,
    }
    result_path = settings.output_dir / "results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name="final_submission"):
        mlflow.log_param("selected_model", selected)
        mlflow.log_metric("selected_oof_logloss", scores[selected])
        mlflow.log_artifact(str(result_path))
        mlflow.log_artifact(str(settings.output_dir / "eda.json"))
        mlflow.log_artifact(str(settings.output_dir / "oof_predictions.parquet"))
        mlflow.log_artifacts(str(models), artifact_path="models")
    return results


def main() -> None:
    """читает параметры командной строки и запускает эксперимент."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "v2")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--svd-iterations", type=int, default=600)
    parser.add_argument("--svd-components", type=int, default=48)
    parser.add_argument("--svd-weight", type=float, default=0.75)
    parser.add_argument("--text-weight", type=float, default=0.05)
    args = parser.parse_args()
    settings = Settings(**vars(args))
    print(json.dumps(run(settings), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
