import json
import time
from pathlib import Path
from urllib.request import urlopen, Request


OUTPUT_DIR = Path("data/raw/ecfr")
TITLES_API = "https://www.ecfr.gov/api/versioner/v1/titles.json"


def get_titles():
    """Get the list of eCFR titles."""

    print("Getting eCFR title list...")

    request = Request(
        TITLES_API,
        headers={
            "User-Agent": "Legal-Agentic-AI-Research/1.0"
        }
    )

    with urlopen(request) as response:
        data = json.load(response)

    titles = data["titles"]

    print(f"Found {len(titles)} titles.")

    return titles


def get_latest_version(title_number):
    """Get the latest available version date for a title."""

    url = (
        f"https://www.ecfr.gov/api/versioner/v1/"
        f"versions/title-{title_number}.json"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Legal-Agentic-AI-Research/1.0"
        }
    )

    with urlopen(request) as response:
        data = json.load(response)

    versions = data.get("content_versions", [])

    if not versions:
        return None

    dates = [
        item["date"]
        for item in versions
        if item.get("date")
    ]

    if not dates:
        return None

    return max(dates)


def download_title(title_number, version_date):
    """Download one complete eCFR title as XML."""

    url = (
        f"https://www.ecfr.gov/api/versioner/v1/full/"
        f"{version_date}/title-{title_number}.xml"
    )

    print(f"Downloading Title {title_number}...")
    print(f"Version date: {version_date}")

    request = Request(
        url,
        headers={
            "User-Agent": "Legal-Agentic-AI-Research/1.0"
        }
    )

    try:

        with urlopen(request) as response:

            data = response.read()

        output_file = (
            OUTPUT_DIR /
            f"title_{title_number}.xml"
        )

        output_file.write_bytes(data)

        print(
            f"Saved: {output_file} "
            f"({len(data):,} bytes)"
        )

        return True

    except Exception as error:

        print(
            f"Failed to download Title {title_number}: "
            f"{error}"
        )

        return False


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    titles = get_titles()

    successful = 0
    failed = 0

    print("\nStarting full eCFR download...")
    print("=" * 60)

    for title in titles:

        title_number = title["number"]

        # Skip reserved titles
        if title.get("reserved", False):

            print(
                f"\nTitle {title_number} "
                f"is reserved. Skipping."
            )

            continue

        try:

            version_date = get_latest_version(
                title_number
            )

            if version_date is None:

                print(
                    f"\nNo version found for "
                    f"Title {title_number}."
                )

                failed += 1
                continue

            success = download_title(
                title_number,
                version_date
            )

            if success:
                successful += 1
            else:
                failed += 1

        except Exception as error:

            print(
                f"\nError processing Title "
                f"{title_number}: {error}"
            )

            failed += 1

        # Small delay between requests
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(
        f"\nFiles are stored in:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()