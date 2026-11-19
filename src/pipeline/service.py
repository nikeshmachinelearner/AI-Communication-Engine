from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from src.monitoring.metrics import (
    COMPONENT_ERRORS,
    COMPONENT_LATENCY,
    MODEL_AVAILABLE,
)


CLASSIFIER_MODEL_PATH = Path(
    "models/classifier/tfidf_logistic.joblib"
)

RETRIEVAL_INDEX_PATH = Path(
    "models/embeddings/messages.faiss"
)

RANKING_MODEL_PATH = Path(
    "models/ranking/lightgbm_ranker.joblib"
)


class CommunicationIntelligenceService:
    def __init__(self) -> None:
        self.classifier = None
        self.retriever = None
        self.extractor = None
        self.ranker = None

        self.loading_errors: dict[str, str] = {}

        self._load_components()

    def _load_components(self) -> None:
        self._load_classifier()
        self._load_retriever()
        self._load_extractor()
        self._load_ranker()

    def _load_classifier(self) -> None:
        try:
            if not CLASSIFIER_MODEL_PATH.exists():
                raise FileNotFoundError(
                    str(CLASSIFIER_MODEL_PATH)
                )

            from src.classification.predict import load_model

            self.classifier = load_model()
            MODEL_AVAILABLE.labels(
                component="classification"
            ).set(1)

        except Exception as error:
            self.loading_errors["classification"] = str(error)
            MODEL_AVAILABLE.labels(
                component="classification"
            ).set(0)

    def _load_retriever(self) -> None:
        try:
            if not RETRIEVAL_INDEX_PATH.exists():
                raise FileNotFoundError(
                    str(RETRIEVAL_INDEX_PATH)
                )

            from src.retrieval.search import ContextRetriever

            self.retriever = ContextRetriever()
            MODEL_AVAILABLE.labels(
                component="retrieval"
            ).set(1)

        except Exception as error:
            self.loading_errors["retrieval"] = str(error)
            MODEL_AVAILABLE.labels(
                component="retrieval"
            ).set(0)

    def _load_extractor(self) -> None:
        try:
            from src.extraction.extractor import (
                InformationExtractor,
            )

            self.extractor = InformationExtractor()
            MODEL_AVAILABLE.labels(
                component="extraction"
            ).set(1)

        except Exception as error:
            self.loading_errors["extraction"] = str(error)
            MODEL_AVAILABLE.labels(
                component="extraction"
            ).set(0)

    def _load_ranker(self) -> None:
        try:
            if not RANKING_MODEL_PATH.exists():
                raise FileNotFoundError(
                    str(RANKING_MODEL_PATH)
                )

            from src.ranking.rank import MessageRanker

            self.ranker = MessageRanker()
            MODEL_AVAILABLE.labels(
                component="ranking"
            ).set(1)

        except Exception as error:
            self.loading_errors["ranking"] = str(error)
            MODEL_AVAILABLE.labels(
                component="ranking"
            ).set(0)

    def status(self) -> dict[str, bool]:
        return {
            "classification": self.classifier is not None,
            "retrieval": self.retriever is not None,
            "extraction": self.extractor is not None,
            "ranking": self.ranker is not None,
        }

    @staticmethod
    def _timed_call(
        component: str,
        function: Any,
        **kwargs: Any,
    ) -> Any:
        start_time = perf_counter()

        try:
            return function(**kwargs)
        except Exception:
            COMPONENT_ERRORS.labels(
                component=component
            ).inc()
            raise
        finally:
            elapsed_seconds = perf_counter() - start_time

            COMPONENT_LATENCY.labels(
                component=component
            ).observe(elapsed_seconds)

    def _classify(self, text: str) -> dict | None:
        if self.classifier is None:
            return None

        pipeline = self.classifier["pipeline"]
        labels = self.classifier["labels"]
        threshold = self.classifier.get("threshold", 0.5)

        probabilities = pipeline.predict_proba([text])[0]

        scores = {
            label: round(float(probability), 4)
            for label, probability in zip(
                labels,
                probabilities,
            )
        }

        predicted_labels = [
            label
            for label, score in scores.items()
            if score >= threshold
        ]

        if not predicted_labels and scores:
            predicted_labels = [
                max(scores, key=scores.get)
            ]

        return {
            "predicted_labels": predicted_labels,
            "scores": scores,
            "threshold": threshold,
        }

    def analyze(
        self,
        text: str,
        reference_time: datetime | None = None,
        timezone_name: str = "Asia/Kathmandu",
        top_k: int = 5,
        conversation_id: str | None = None,
        sender_importance: float = 0.5,
        conversation_relevance: float = 0.5,
        hours_until_deadline: float | None = None,
    ) -> dict:
        total_start = perf_counter()

        component_errors: dict[str, str] = {}

        classification = None
        extraction = None
        context = None
        ranking = None

        if self.classifier is not None:
            try:
                classification = self._timed_call(
                    component="classification",
                    function=self._classify,
                    text=text,
                )
            except Exception as error:
                component_errors["classification"] = str(error)

        if self.extractor is not None:
            try:
                extraction = self._timed_call(
                    component="extraction",
                    function=self.extractor.extract,
                    text=text,
                    reference_time=reference_time,
                    timezone_name=timezone_name,
                )
            except Exception as error:
                component_errors["extraction"] = str(error)

        if self.retriever is not None:
            try:
                context = self._timed_call(
                    component="retrieval",
                    function=self.retriever.search,
                    query=text,
                    top_k=top_k,
                    conversation_id=conversation_id,
                )
            except Exception as error:
                component_errors["retrieval"] = str(error)

        if self.ranker is not None:
            try:
                ranking_input = [
                    {
                        "message_id": "incoming_message",
                        "text": text,
                        "sender_importance": sender_importance,
                        "conversation_relevance": (
                            conversation_relevance
                        ),
                        "hours_until_deadline": (
                            hours_until_deadline
                        ),
                    }
                ]

                ranking = self._timed_call(
                    component="ranking",
                    function=self.ranker.rank,
                    messages=ranking_input,
                )
            except Exception as error:
                component_errors["ranking"] = str(error)

        recommended_actions = self._recommend_actions(
            classification=classification,
            extraction=extraction,
        )

        total_latency_ms = (
            perf_counter() - total_start
        ) * 1000

        return {
            "text": text,
            "classification": classification,
            "extraction": extraction,
            "context": context,
            "ranking": ranking,
            "recommended_actions": recommended_actions,
            "component_errors": component_errors,
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),
        }

    @staticmethod
    def _recommend_actions(
        classification: dict | None,
        extraction: dict | None,
    ) -> list[str]:
        actions = []

        labels = set()

        if classification:
            labels = set(
                classification.get(
                    "predicted_labels",
                    [],
                )
            )

        if extraction:
            for task in extraction.get("tasks", []):
                task_text = task.get("text")

                if task_text:
                    actions.append(
                        f"Complete task: {task_text}"
                    )

            for commitment in extraction.get(
                "commitments",
                [],
            ):
                commitment_text = commitment.get("text")

                if commitment_text:
                    actions.append(
                        "Track commitment: "
                        f"{commitment_text}"
                    )

        if "meeting_request" in labels:
            actions.append(
                "Review availability and schedule the meeting."
            )

        if "question" in labels:
            actions.append(
                "Prepare a response to the sender."
            )

        if "urgent" in labels:
            actions.append(
                "Review this message immediately."
            )

        deduplicated_actions = []
        seen = set()

        for action in actions:
            normalized = action.lower()

            if normalized not in seen:
                deduplicated_actions.append(action)
                seen.add(normalized)

        return deduplicated_actions