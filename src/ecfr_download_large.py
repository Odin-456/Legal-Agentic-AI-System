import json
import requests
from pathlib import Path
import time

BASE_URL = "https://www.ecfr.gov/api/versioner/v1"

DATE = "2025-12-01"
TITLE = "7"

OUTPUT_DIR = Path("data/raw/ecfr/title-7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_structure(title):
    url = f"{BASE_URL}/structure/{DATE}/title-{title}.json"

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    return response.json()


def extract_parts(node, parts=None):
    if parts is None:
        parts = []

    if node.get("type") == "part":
        parts.append({
            "identifier": node["identifier"],
            "label": node.get("label"),
            "size": node.get("size", 0),
        })

    for child in node.get("children", []):
        extract_parts(child, parts)

    return parts


def download_part(title, part):
    part_id = part["identifier"]

    output_file = OUTPUT_DIR / f"title-{title}-part-{part_id}.xml"

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"SKIP  Part {part_id} - already exists")
        return True

    url = (
        f"{BASE_URL}/full/{DATE}/"
        f"title-{title}.xml?part={part_id}"
    )

    print(f"DOWN  Part {part_id}: {part['label']}")

    try:
        response = requests.get(
            url,
            timeout=300
        )

        response.raise_for_status()

        content = response.content

        if len(content) == 0:
            print(f"FAIL  Part {part_id}: empty response")
            return False

        output_file.write_bytes(content)

        print(
            f"OK    Part {part_id}: "
            f"{len(content):,} bytes"
        )

        return True

    except Exception as e:
        print(f"FAIL  Part {part_id}: {e}")
        return False


def main():

    print("=" * 60)
    print(f"Downloading eCFR Title {TITLE}")
    print("=" * 60)

    structure = get_structure(TITLE)

    parts = extract_parts(structure)

    print(f"Total parts found: {len(parts)}")

    # Remove reserved / zero-size parts
    active_parts = [
        p for p in parts
        if p["size"] > 0
    ]

    print(f"Active parts: {len(active_parts)}")
    print(
        f"Reserved/empty parts skipped: "
        f"{len(parts) - len(active_parts)}"
    )

    successful = 0
    failed = []

    for i, part in enumerate(active_parts, start=1):

        print(
            f"\n[{i}/{len(active_parts)}]"
        )

        success = download_part(TITLE, part)

        if success:
            successful += 1
        else:
            failed.append(part["identifier"])

        # Be polite to the API
        time.sleep(0.2)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        failure_file = Path(
            "data/raw/ecfr/title-7-failures.json"
        )

        failure_file.write_text(
            json.dumps(failed, indent=2)
        )

        print(f"Failure log: {failure_file}")


if __name__ == "__main__":
    main()