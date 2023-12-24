# Huggingface 🤗 Datasets

Notes on using Huggingface Datasets

- [Huggingface 🤗 Datasets](#huggingface--datasets)
  - [Source data](#source-data)
    - [Caching](#caching)
  - [Standard Preprocessing](#standard-preprocessing)
  - [Selected datasets](#selected-datasets)
    - [bookcorpus](#bookcorpus)
    - [c4 - realnewslike](#c4---realnewslike)
    - [wikimedia/wikipedia - english](#wikimediawikipedia---english)
  - [Sentence Splitting](#sentence-splitting)

## Source data

Huggingface's `datasets` package provides easy access to predefined data.

These are type [`datasets.arrow_dataset.Dataset`](https://huggingface.co/docs/datasets/main/en/about_arrow) --
loading these creates a memory map to on-disk cache but does not _actually_ load files into memory.

Transformations can be applied to datasets using `map()` or `..._transform()` methods.
`map` will apply eagerly to standard Datasets and lazily to IterableDatasets.
`with_transform()` applies eagerly, while `set_transform()` is applied lazily.

### Caching

Caching of data and prior work can be managed with [`download_mode` kwarg](https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/builder_classes#datasets.DownloadMode) of `datasets.load_dataset(..., download_mode=...)`.

`download_mode` | Downloads | Dataset
--- | --- | ---
`reuse_dataset_if_exists` (default) | Reuse | Reuse
`reuse_cache_if_exists` | Reuse | Fresh
`force_redownload` | Fresh | Fresh

We can also leverage dataset functionality like map() and filter() to apply batched transformations
or even use an IterableDataset to lazily apply transformations as needed

## Standard Preprocessing

A preprocessing pipeline relies on [Huggingface Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-tokenizer-block-by-block) to efficiently apply standard text cleaning and normalization steps.

- normalize - clean accents, convert to ASCII, lowercase, remove tags, and normalize whitespace
- split text blobs to the sentence level

## Selected datasets

### [bookcorpus](https://huggingface.co/datasets/bookcorpus)

Bookcorpus contains sentence-level data from various books.

In additional to the standard preprocessing, we revert the existing preprocessing by NLTK's TreebankWordTokenize.
`bookcorpus` text has sentence-level granularity; we do not split further in preprocessing.

### [c4 - realnewslike](https://huggingface.co/datasets/c4)

`c4` is "A colossal, cleaned version of Common Crawl's web crawl corpus. Based on Common Crawl dataset: "<https://commoncrawl.org>"."

The `realnewslike` subset applies source filters to only include content from the domains used in the 'RealNews' dataset (Zellers et al., 2019).

Standard preprocessing is applied.

### [wikimedia/wikipedia - english](https://huggingface.co/datasets/wikimedia/wikipedia)

The `wikipedia` dataset is built from the [Wikipedia dumps](https://dumps.wikimedia.org/) with one subset per language, each containing a single train split.

Dataset articles appear to include extraneous appendices that are not good examples of natural language.  As a result, we truncate the main article blob to not include headings included in the [standard appendicdes and footers](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout#Standard_appendices_and_footers).

Once truncated, standard preprocessing is applied.

## Sentence Splitting

Edge cases in sentence splitting can be tested with:

<!-- markdownlint-disable MD013 -->
```txt
Dr. Smith, who works at the University of California, Berkeley, sent me an email. He said, "I don't understand why we can't apply for a grant." I replied, "Maybe we should contact the university administration for assistance."

# ref https://gist.github.com/owens2727/b936168921d3468d88bb27d2016044c9
He removed the director! Of the FBI? You're kidding... Mr. Comey was removed from his post as F.B.I. Director in May of 2017\nMr. Comey used to the Director of the F.B.I. Now he is a private citizen. James B. Comey was born on December 14, 1960. In college, James Comey never once earned a B.He got straight A's.
```
<!-- markdownlint-enable -->
