# NLP Comparative Analysis Project

## Purpose

This project uses Natural Language Processing (NLP) with NLTK to compare three related texts and one unknown text sample. It performs tokenization, stemming, lemmatization, named entity recognition, and trigram analysis so the user can:

- identify the most common language patterns in each text
- count named entities in each text
- infer the shared subject of the first three texts
- compare stylistic trigram patterns to estimate which author style is closest to the fourth text

## Project Files

- [NLTK Assignment](</c:/Users/camjh/Downloads/INFO 2/NLTK Assignment>) contains the full Python implementation
- [analysis_results.txt](</c:/Users/camjh/Downloads/INFO 2/analysis_results.txt>) is generated when the script runs
- [RJ_Tolkein.txt](</c:/Users/camjh/Downloads/INFO 2/RJ_Tolkein.txt>) is Text 1
- [RJ_Martin.txt](</c:/Users/camjh/Downloads/INFO 2/RJ_Martin.txt>) is Text 2
- [RJ_Lovecraft.txt](</c:/Users/camjh/Downloads/INFO 2/RJ_Lovecraft.txt>) is Text 3
- [Martin.txt](</c:/Users/camjh/Downloads/INFO 2/Martin.txt>) is used as Text 4 for authorship comparison

## Class Design

The implementation uses two classes to keep the project organized.

### `TextAnalyzer`

This class is responsible for analyzing one text file at a time.

#### Attributes

- `label`: a readable name for the text being analyzed
- `file_path`: the path to the source text file
- `raw_text`: the original file contents
- `word_tokens`: the tokenized text before filtering
- `alpha_tokens`: lowercase alphabetic tokens only
- `content_tokens`: alphabetic tokens with English stopwords removed
- `stemmed_tokens`: stemmed version of the filtered tokens
- `lemmatized_tokens`: lemmatized version of the filtered tokens
- `named_entities`: named entity mentions extracted by NLTK
- `trigram_counts`: a `Counter` containing all trigram frequencies
- `stop_words`: the English stopword set used for filtering
- `stemmer`: a `PorterStemmer` instance
- `lemmatizer`: a `WordNetLemmatizer` instance

#### Methods

- `load_text()`: reads the text file into memory
- `tokenize_text()`: tokenizes the file and creates cleaned token lists
- `stem_tokens()`: applies stemming to the filtered tokens
- `lemmatize_tokens()`: applies lemmatization using part-of-speech tags
- `extract_named_entities()`: finds named entity mentions with `ne_chunk`
- `build_trigrams()`: creates trigram frequencies from the token stream
- `analyze()`: runs the full pipeline and returns a structured result object
- `_wordnet_pos(tag)`: maps NLTK POS tags to WordNet POS values for better lemmatization

### `NLPComparisonProject`

This class manages the full assignment workflow across all texts.

#### Attributes

- `texts`: a dictionary of the first three labeled source files
- `unknown_text`: the labeled fourth text used for authorship comparison
- `results`: stores the finished results for the first three texts
- `unknown_result`: stores the finished result for Text 4

#### Methods

- `run()`: analyzes all texts and writes the final report
- `build_report()`: assembles the printable report text
- `determine_subject()`: uses shared named entities and high-frequency lemmas to infer the subject
- `compare_unknown_text()`: compares trigram overlap between Text 4 and the first three texts
- `_format_counter(...)`: formats token, stem, and lemma counts for the report
- `_format_trigrams(...)`: formats trigram counts for the report
- `_format_result_section(...)`: formats one text analysis section for output

## Implementation Notes

- The script automatically downloads missing NLTK resources the first time it runs.
- Token frequency counts are based on lowercase alphabetic words with English stopwords removed so the results highlight meaningful content instead of filler words such as `the`, `and`, or `of`.
- Trigrams are built from lowercase alphabetic tokens and keep stopwords when helpful, because short phrase patterns are useful for authorship-style comparisons.
- Named entities are counted as entity mentions, and the report also includes the unique entity list for context.

## How To Run

Install the required packages if they are not already available:

```powershell
py -m pip install --user nltk numpy
```

Use the Windows Python launcher:

```powershell
py "NLTK Assignment"
```

After the script finishes, it prints the report to the console and saves the same output to `analysis_results.txt`.

## Limitations

- NLTK named entity recognition is rule-based and may miss fictional names or classify them inconsistently.
- The authorship conclusion is an inference based on trigram overlap, not a definitive proof of authorship.
- Because the fourth text is longer than the rewritten Romeo and Juliet samples, its broader vocabulary can reduce direct trigram overlap.
- Stopword removal improves interpretability for common-token analysis, but it also changes the exact token counts compared with a fully unfiltered corpus.
- The analysis works best on English prose and would need adjustment for other languages or text genres.
