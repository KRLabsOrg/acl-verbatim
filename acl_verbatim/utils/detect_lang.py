import argparse
import json

from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Detect paper language from title")
    parser.add_argument("--input-file", required=True, help="Input file")
    parser.add_argument("--output-file", required=True, help="Output file")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input_file) as f:
        papers = json.load(f)
    print(f"will detect the language of {len(papers)} papers (from the titles)")
    languages = [Language.ENGLISH, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    for i, paper in tqdm(enumerate(papers)):
        lang = detector.detect_language_of(paper["title"])
        if lang is None:
            paper["lang"] = "other"
        else:
            paper["lang"] = lang.iso_code_639_3.name.lower()
        if i > 1000:
            break
    with open(args.output_file, "w") as of:
        json.dump(papers, of)


if __name__ == "__main__":
    main()
