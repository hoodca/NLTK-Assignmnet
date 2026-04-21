from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Iterable

import nltk
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "analysis_results.txt"
TOP_N = 20
TOP_TRIGRAMS = 15

# These files map directly to the assignment's three known comparison texts.
TEXT_FILES = {
    "Text 1 - Tolkien Style": BASE_DIR / "RJ_Tolkein.txt",
    "Text 2 - Martin Style": BASE_DIR / "RJ_Martin.txt",
    "Text 3 - Lovecraft Style": BASE_DIR / "RJ_Lovecraft.txt",
}

TEXT_4 = ("Text 4 - Unknown Author", BASE_DIR / "Martin.txt")

# NLTK looks up resources by internal folder name, not just the download package name.
RESOURCE_PATHS = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words",
}


@dataclass
class AnalysisResult:
    label: str
    file_name: str
    top_tokens: list[tuple[str, int]]
    top_stems: list[tuple[str, int]]
    top_lemmas: list[tuple[str, int]]
    named_entity_count: int
    unique_named_entities: list[str]
    top_trigrams: list[tuple[tuple[str, str, str], int]]
    all_trigram_counts: Counter


class TextAnalyzer:
    """Analyze one text file with NLTK token, entity, and n-gram tooling."""

    def __init__(self, label: str, file_path: Path) -> None:
        self.label = label
        self.file_path = file_path
        self.raw_text = ""
        self.word_tokens: list[str] = []
        self.alpha_tokens: list[str] = []
        self.content_tokens: list[str] = []
        self.stemmed_tokens: list[str] = []
        self.lemmatized_tokens: list[str] = []
        self.named_entities: list[str] = []
        self.trigram_counts: Counter = Counter()
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def load_text(self) -> None:
        self.raw_text = self.file_path.read_text(encoding="utf-8")

    def tokenize_text(self) -> None:
        self.word_tokens = word_tokenize(self.raw_text)
        self.alpha_tokens = [token.lower() for token in self.word_tokens if token.isalpha()]
        # Remove common filler words so the frequency lists better reflect subject matter.
        self.content_tokens = [
            token for token in self.alpha_tokens if token not in self.stop_words and len(token) > 1
        ]

    def stem_tokens(self) -> None:
        self.stemmed_tokens = [self.stemmer.stem(token) for token in self.content_tokens]

    def lemmatize_tokens(self) -> None:
        tagged_tokens = pos_tag(self.content_tokens)
        # Lemmatization is more accurate when we convert NLTK POS tags to WordNet tags first.
        self.lemmatized_tokens = [
            self.lemmatizer.lemmatize(token, self._wordnet_pos(tag))
            for token, tag in tagged_tokens
        ]

    def extract_named_entities(self) -> None:
        entities: list[str] = []
        # Named entity chunking works on POS-tagged sentences, so we process one sentence at a time.
        for sentence in sent_tokenize(self.raw_text):
            sentence_tokens = word_tokenize(sentence)
            tagged_sentence = pos_tag(sentence_tokens)
            chunked_sentence = ne_chunk(tagged_sentence)
            for chunk in chunked_sentence:
                if hasattr(chunk, "label"):
                    entity = " ".join(token for token, _ in chunk.leaves())
                    entities.append(entity)
        self.named_entities = entities

    def build_trigrams(self) -> None:
        trigram_stream = ngrams(self.alpha_tokens, 3)
        # Keep trigrams that contain at least one meaningful word so phrase patterns remain readable.
        meaningful_trigrams = [gram for gram in trigram_stream if any(word not in self.stop_words for word in gram)]
        self.trigram_counts = Counter(meaningful_trigrams)

    def analyze(self) -> AnalysisResult:
        self.load_text()
        self.tokenize_text()
        self.stem_tokens()
        self.lemmatize_tokens()
        self.extract_named_entities()
        self.build_trigrams()

        return AnalysisResult(
            label=self.label,
            file_name=self.file_path.name,
            top_tokens=Counter(self.content_tokens).most_common(TOP_N),
            top_stems=Counter(self.stemmed_tokens).most_common(TOP_N),
            top_lemmas=Counter(self.lemmatized_tokens).most_common(TOP_N),
            named_entity_count=len(self.named_entities),
            unique_named_entities=sorted(set(self.named_entities)),
            top_trigrams=self.trigram_counts.most_common(TOP_TRIGRAMS),
            all_trigram_counts=self.trigram_counts,
        )

    @staticmethod
    def _wordnet_pos(tag: str) -> str:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("N"):
            return "n"
        if tag.startswith("R"):
            return "r"
        return "n"


class NLPComparisonProject:
    """Coordinate analysis across the three known texts and the unknown sample."""

    def __init__(self, texts: dict[str, Path], unknown_text: tuple[str, Path]) -> None:
        self.texts = texts
        self.unknown_text = unknown_text
        self.results: dict[str, AnalysisResult] = {}
        self.unknown_result: AnalysisResult | None = None

    def run(self) -> str:
        for label, file_path in self.texts.items():
            self.results[label] = TextAnalyzer(label, file_path).analyze()

        unknown_label, unknown_path = self.unknown_text
        self.unknown_result = TextAnalyzer(unknown_label, unknown_path).analyze()

        report = self.build_report()
        REPORT_PATH.write_text(report, encoding="utf-8")
        return report

    def build_report(self) -> str:
        sections = [
            "NLP Comparative Analysis Report",
            "=" * 31,
            "",
            "Files analyzed:",
        ]
        sections.extend(f"- {result.label}: {result.file_name}" for result in self.results.values())
        if self.unknown_result is not None:
            sections.append(f"- {self.unknown_result.label}: {self.unknown_result.file_name}")
        sections.append("")

        for result in self.results.values():
            sections.extend(self._format_result_section(result))

        sections.append("Shared Subject Inference")
        sections.append("-" * 24)
        sections.append(self.determine_subject())
        sections.append("")

        if self.unknown_result is not None:
            sections.extend(self._format_result_section(self.unknown_result))
            sections.append("Authorship Inference For Text 4")
            sections.append("-" * 30)
            sections.append(self.compare_unknown_text())
            sections.append("")

        return "\n".join(sections)

    def determine_subject(self) -> str:
        # Intersect entity and lemma sets to find clues that remain consistent across all three texts.
        entity_sets = [
            {entity.lower() for entity in result.unique_named_entities}
            for result in self.results.values()
        ]
        shared_entities = reduce(set.intersection, entity_sets) if entity_sets else set()

        lemma_sets = [
            {lemma for lemma, _ in result.top_lemmas}
            for result in self.results.values()
        ]
        shared_lemmas = reduce(set.intersection, lemma_sets) if lemma_sets else set()

        evidence = []
        for candidate in ("romeo", "juliet", "verona", "capulet", "montague"):
            if candidate in shared_entities or candidate in shared_lemmas:
                evidence.append(candidate.title())

        if evidence:
            return (
                "The three main texts are all retellings of Romeo and Juliet. "
                f"The conclusion is supported by shared high-value terms and named entities such as: {', '.join(evidence)}."
            )

        if shared_entities:
            entity_list = ", ".join(sorted(shared_entities))
            return (
                "The three main texts appear to share the same subject because they reuse the same core named entities: "
                f"{entity_list}."
            )

        return (
            "The three texts share a common tragic romance subject, but the strongest indicators are distributed "
            "across the token and entity lists rather than a single exact overlap."
        )

    def compare_unknown_text(self) -> str:
        if self.unknown_result is None:
            return "No fourth text was analyzed."

        comparisons = []

        for label, result in self.results.items():
            # Shared trigram counts give a simple style signal without relying on exact full-text matches.
            shared_trigrams = sorted(
                (
                    (
                        min(result.all_trigram_counts[gram], self.unknown_result.all_trigram_counts[gram]),
                        gram,
                    )
                    for gram in result.all_trigram_counts
                    if gram in self.unknown_result.all_trigram_counts
                ),
                reverse=True,
            )
            shared_type_count = len(shared_trigrams)
            weighted_overlap = sum(
                min(result.all_trigram_counts[gram], self.unknown_result.all_trigram_counts[gram])
                for gram in result.all_trigram_counts
                if gram in self.unknown_result.all_trigram_counts
            )

            author_set = set(result.all_trigram_counts)
            unknown_set = set(self.unknown_result.all_trigram_counts)
            jaccard = len(author_set & unknown_set) / len(author_set | unknown_set)

            shared_examples = [f"{' '.join(gram)} ({score})" for score, gram in shared_trigrams[:5]]
            comparisons.append((label, shared_type_count, weighted_overlap, jaccard, shared_examples))

        comparisons.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        best_match = comparisons[0]

        lines = [
            f"The strongest trigram match for {self.unknown_result.file_name} is {best_match[0]}.",
            "",
            "Similarity scores:",
        ]

        for label, shared_type_count, weighted_overlap, jaccard, shared_examples in comparisons:
            lines.append(
                f"- {label}: shared trigram types = {shared_type_count}, weighted overlap = {weighted_overlap}, "
                f"Jaccard similarity = {jaccard:.4f}"
            )
            if shared_examples:
                lines.append(f"  Shared examples: {', '.join(shared_examples)}")

        lines.append("")
        lines.append(
            "Based on the trigram overlap, the unknown text was most likely written in the style of the author "
            f"represented by {best_match[0]}."
        )

        return "\n".join(lines)

    @staticmethod
    def _format_counter(counter_items: Iterable[tuple[str, int]]) -> str:
        return ", ".join(f"{item} ({count})" for item, count in counter_items)

    @staticmethod
    def _format_trigrams(trigram_items: Iterable[tuple[tuple[str, str, str], int]]) -> str:
        return ", ".join(f"{' '.join(gram)} ({count})" for gram, count in trigram_items)

    def _format_result_section(self, result: AnalysisResult) -> list[str]:
        preview_entities = ", ".join(result.unique_named_entities[:10]) if result.unique_named_entities else "None"
        if len(result.unique_named_entities) > 10:
            preview_entities += ", ..."

        return [
            result.label,
            "-" * len(result.label),
            f"File: {result.file_name}",
            f"Top {TOP_N} tokens: {self._format_counter(result.top_tokens)}",
            f"Top {TOP_N} stems: {self._format_counter(result.top_stems)}",
            f"Top {TOP_N} lemmas: {self._format_counter(result.top_lemmas)}",
            f"Named entity mentions: {result.named_entity_count}",
            f"Unique named entities ({len(result.unique_named_entities)}): {preview_entities}",
            f"Top {TOP_TRIGRAMS} trigrams: {self._format_trigrams(result.top_trigrams)}",
            "",
        ]


def ensure_nltk_resources() -> None:
    # Download only what is missing so repeated runs stay fast.
    for package, resource_path in RESOURCE_PATHS.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


def main() -> None:
    ensure_nltk_resources()
    project = NLPComparisonProject(TEXT_FILES, TEXT_4)
    report = project.run()
    print(report)
    print(f"\nReport saved to: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
