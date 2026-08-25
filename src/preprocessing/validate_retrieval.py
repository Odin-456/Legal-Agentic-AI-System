import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/retrieval/title-7")


# ============================================================
# VALIDATE ONE JSON FILE
# ============================================================

def validate_file(json_file):
    """
    Validate one retrieval-ready JSON file.

    Returns:
        documents: number of documents
        errors: list of validation errors
    """

    errors = []

    try:
        # ----------------------------------------------------
        # Load JSON
        # ----------------------------------------------------

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:

        return 0, [
            f"Could not read JSON: {e}"
        ]

    # --------------------------------------------------------
    # JSON ROOT MUST BE A LIST
    # --------------------------------------------------------

    if not isinstance(data, list):

        return 0, [
            "Root JSON object is not a list."
        ]

    documents = len(data)

    # --------------------------------------------------------
    # CHECK EACH DOCUMENT
    # --------------------------------------------------------

    for index, document in enumerate(data):

        location = f"{json_file.name} document {index + 1}"

        # ----------------------------------------------------
        # DOCUMENT MUST BE A DICTIONARY
        # ----------------------------------------------------

        if not isinstance(document, dict):

            errors.append(
                f"{location}: document is not an object."
            )

            continue

        # ----------------------------------------------------
        # REQUIRED TOP-LEVEL FIELDS
        # ----------------------------------------------------

        if "text" not in document:

            errors.append(
                f"{location}: missing 'text'."
            )

        if "retrieval_text" not in document:

            errors.append(
                f"{location}: missing 'retrieval_text'."
            )

        if "metadata" not in document:

            errors.append(
                f"{location}: missing 'metadata'."
            )

            continue

        # ----------------------------------------------------
        # CHECK TEXT
        # ----------------------------------------------------

        text = document.get("text")

        if not isinstance(text, str):

            errors.append(
                f"{location}: 'text' is not a string."
            )

        elif not text.strip():

            errors.append(
                f"{location}: 'text' is empty."
            )

        # ----------------------------------------------------
        # CHECK RETRIEVAL TEXT
        # ----------------------------------------------------

        retrieval_text = document.get(
            "retrieval_text"
        )

        if not isinstance(
            retrieval_text,
            str
        ):

            errors.append(
                f"{location}: "
                "'retrieval_text' is not a string."
            )

        elif not retrieval_text.strip():

            errors.append(
                f"{location}: "
                "'retrieval_text' is empty."
            )

        # ----------------------------------------------------
        # CHECK METADATA
        # ----------------------------------------------------

        metadata = document.get("metadata")

        if not isinstance(
            metadata,
            dict
        ):

            errors.append(
                f"{location}: "
                "'metadata' is not an object."
            )

            continue

        # ----------------------------------------------------
        # REQUIRED METADATA FIELDS
        # ----------------------------------------------------

        required_metadata = [
            "title",
            "part",
            "part_title",
            "section",
            "section_title",
            "paragraph",
            "subparagraph",
            "clause",
            "subclause",
            "citation",
            "source_file",
        ]

        for field in required_metadata:

            if field not in metadata:

                errors.append(
                    f"{location}: "
                    f"missing metadata '{field}'."
                )

        # ----------------------------------------------------
        # CITATION MUST EXIST
        # ----------------------------------------------------

        citation = metadata.get("citation")

        if citation is not None:

            if not isinstance(
                citation,
                str
            ):

                errors.append(
                    f"{location}: "
                    "'citation' is not a string."
                )

            elif not citation.strip():

                errors.append(
                    f"{location}: "
                    "'citation' is empty."
                )

        # ----------------------------------------------------
        # RETRIEVAL TEXT SHOULD CONTAIN CITATION
        # ----------------------------------------------------

        if (
            isinstance(retrieval_text, str)
            and isinstance(citation, str)
            and citation
        ):

            if citation not in retrieval_text:

                errors.append(
                    f"{location}: "
                    "citation missing from retrieval_text."
                )

    return documents, errors


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 60)
    print("RETRIEVAL DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # FIND JSON FILES
    # --------------------------------------------------------

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nJSON files found: {len(json_files)}"
    )

    if not json_files:

        print(
            "\nERROR: No JSON files found."
        )

        print(
            f"Expected directory: {INPUT_DIR}"
        )

        return

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    files_checked = 0
    valid_files = 0
    invalid_files = 0

    total_documents = 0
    total_errors = 0

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for number, json_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\rProcessing: "
            f"{number}/{len(json_files)}",
            end="",
            flush=True
        )

        documents, errors = validate_file(
            json_file
        )

        files_checked += 1
        total_documents += documents

        if errors:

            invalid_files += 1
            total_errors += len(errors)

            print(
                f"\n\nERRORS IN: "
                f"{json_file.name}"
            )

            for error in errors[:10]:

                print(
                    f"  - {error}"
                )

            if len(errors) > 10:

                print(
                    f"  ... and "
                    f"{len(errors) - 10} "
                    f"more errors."
                )

        else:

            valid_files += 1

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print(
        f"JSON files found:      "
        f"{len(json_files)}"
    )

    print(
        f"Files checked:         "
        f"{files_checked}"
    )

    print(
        f"Valid files:           "
        f"{valid_files}"
    )

    print(
        f"Invalid files:         "
        f"{invalid_files}"
    )

    print(
        f"Total documents:       "
        f"{total_documents}"
    )

    print(
        f"Total errors:          "
        f"{total_errors}"
    )

    # --------------------------------------------------------
    # SUCCESS / FAILURE
    # --------------------------------------------------------

    if total_errors == 0:

        print("\nSUCCESS:")
        print(
            "All retrieval JSON files "
            "passed validation."
        )

    else:

        print("\nWARNING:")
        print(
            "Some retrieval JSON files "
            "contain validation errors."
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()