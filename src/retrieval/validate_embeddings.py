import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "data/processed/embeddings/title-7"
)

EXPECTED_DIMENSION = 768


# ============================================================
# VALIDATE ONE FILE
# ============================================================

def validate_file(input_file):

    errors = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    if not isinstance(documents, list):

        errors.append(
            "Root JSON object is not a list."
        )

        return 0, errors

    for index, document in enumerate(
        documents
    ):

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_fields = [
            "id",
            "embedding",
            "text",
            "retrieval_text",
            "metadata",
        ]

        for field in required_fields:

            if field not in document:

                errors.append(
                    f"Document {index}: "
                    f"missing field '{field}'."
                )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embedding = document.get(
            "embedding"
        )

        if not isinstance(
            embedding,
            list
        ):

            errors.append(
                f"Document {index}: "
                "embedding is not a list."
            )

            continue

        # ----------------------------------------------------
        # Dimension
        # ----------------------------------------------------

        if len(embedding) != (
            EXPECTED_DIMENSION
        ):

            errors.append(
                f"Document {index}: "
                f"expected {EXPECTED_DIMENSION} "
                f"dimensions, got {len(embedding)}."
            )

        # ----------------------------------------------------
        # Numeric values
        # ----------------------------------------------------

        try:

            vector = np.asarray(
                embedding,
                dtype=np.float32
            )

            if not np.all(
                np.isfinite(vector)
            ):

                errors.append(
                    f"Document {index}: "
                    "embedding contains NaN "
                    "or infinite values."
                )

        except Exception as e:

            errors.append(
                f"Document {index}: "
                f"invalid embedding values: {e}"
            )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if not isinstance(
            document.get("text"),
            str
        ):

            errors.append(
                f"Document {index}: "
                "text is not a string."
            )

        # ----------------------------------------------------
        # Retrieval text
        # ----------------------------------------------------

        if not isinstance(
            document.get("retrieval_text"),
            str
        ):

            errors.append(
                f"Document {index}: "
                "retrieval_text is not a string."
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if not isinstance(
            document.get("metadata"),
            dict
        ):

            errors.append(
                f"Document {index}: "
                "metadata is not an object."
            )

    return len(documents), errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("eCFR EMBEDDING VALIDATION")
    print("=" * 60)

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"\nJSON files found:"
        f" {len(json_files)}"
    )

    if not json_files:

        print(
            "\nERROR:"
            " No embedding JSON files found."
        )

        return

    files_checked = 0
    valid_files = 0
    invalid_files = 0
    total_documents = 0
    total_errors = 0

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for number, input_file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\rProcessing:"
            f" {number}/{len(json_files)}",
            end=""
        )

        try:

            documents, errors = validate_file(
                input_file
            )

            files_checked += 1
            total_documents += documents

            if errors:

                invalid_files += 1
                total_errors += len(errors)

                print(
                    f"\n\nERRORS in "
                    f"{input_file.name}:"
                )

                for error in errors[:10]:

                    print(
                        f"  - {error}"
                    )

                if len(errors) > 10:

                    print(
                        f"  ... and "
                        f"{len(errors) - 10} "
                        f"more errors"
                    )

            else:

                valid_files += 1

        except Exception as e:

            files_checked += 1
            invalid_files += 1
            total_errors += 1

            print(
                f"\n\nERROR reading "
                f"{input_file.name}: {e}"
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")

    print("=" * 60)
    print("EMBEDDING VALIDATION COMPLETE")
    print("=" * 60)

    print(
        f"JSON files found:       "
        f"{len(json_files)}"
    )

    print(
        f"Files checked:          "
        f"{files_checked}"
    )

    print(
        f"Valid files:            "
        f"{valid_files}"
    )

    print(
        f"Invalid files:          "
        f"{invalid_files}"
    )

    print(
        f"Total documents:        "
        f"{total_documents}"
    )

    print(
        f"Expected dimensions:    "
        f"{EXPECTED_DIMENSION}"
    )

    print(
        f"Total errors:           "
        f"{total_errors}"
    )

    if (
        invalid_files == 0
        and total_errors == 0
    ):

        print("\nSUCCESS:")
        print(
            "All embedding files passed validation."
        )

    else:

        print("\nWARNING:")
        print(
            "Some embedding files contain errors."
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()