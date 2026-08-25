import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/processed/ecfr/title-7")


# ============================================================
# VALIDATE ONE JSON FILE
# ============================================================

def validate_json_file(json_file):

    errors = []

    try:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

    except Exception as e:

        return [f"Could not read JSON: {e}"]

    # --------------------------------------------------------
    # CHECK ROOT STRUCTURE
    # --------------------------------------------------------

    if not isinstance(documents, list):

        errors.append(
            "Root JSON object is not a list."
        )

        return errors

    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

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

    for index, document in enumerate(documents):

        # ----------------------------------------------------
        # DOCUMENT STRUCTURE
        # ----------------------------------------------------

        if not isinstance(document, dict):

            errors.append(
                f"Document {index} is not a dictionary."
            )

            continue

        if "text" not in document:

            errors.append(
                f"Document {index} missing 'text'."
            )

        if "metadata" not in document:

            errors.append(
                f"Document {index} missing 'metadata'."
            )

            continue

        metadata = document["metadata"]

        # ----------------------------------------------------
        # METADATA STRUCTURE
        # ----------------------------------------------------

        for field in required_metadata:

            if field not in metadata:

                errors.append(
                    f"Document {index} missing "
                    f"metadata field '{field}'."
                )

        # ----------------------------------------------------
        # EMPTY TEXT
        # ----------------------------------------------------

        if not document.get("text", "").strip():

            errors.append(
                f"Document {index} has empty text."
            )

    return errors


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 60)
    print("eCFR JSON VALIDATION")
    print("=" * 60)

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(
        f"JSON files found: {len(json_files)}"
    )

    if not json_files:

        print(
            "ERROR: No JSON files found."
        )

        return

    total_documents = 0
    total_errors = 0
    valid_files = 0
    invalid_files = []

    # --------------------------------------------------------
    # VALIDATE EVERY FILE
    # --------------------------------------------------------

    for index, json_file in enumerate(
        json_files,
        start=1
    ):

        errors = validate_json_file(
            json_file
        )

        # Count documents separately
        try:

            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as f:

                documents = json.load(f)

            if isinstance(documents, list):

                total_documents += len(documents)

        except Exception:
            pass

        if errors:

            invalid_files.append(
                (
                    json_file.name,
                    errors
                )
            )

            total_errors += len(errors)

            print(
                f"[INVALID] {json_file.name}"
            )

            for error in errors[:5]:

                print(
                    f"    {error}"
                )

        else:

            valid_files += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    print(
        f"JSON files checked:    {len(json_files)}"
    )

    print(
        f"Valid files:           {valid_files}"
    )

    print(
        f"Invalid files:         {len(invalid_files)}"
    )

    print(
        f"Total documents:       {total_documents}"
    )

    print(
        f"Total errors:          {total_errors}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if not invalid_files:

        print("\nSUCCESS:")
        print(
            "All JSON files passed validation."
        )

    else:

        print("\nWARNING:")
        print(
            "Some JSON files contain errors."
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()