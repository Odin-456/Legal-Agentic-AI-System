import time
from pathlib import Path
from urllib.request import urlopen, Request
import json


OUTPUT_DIR = Path("data/raw/ecfr")

FAILED_TITLES = [35]


def get_latest_version(title_number):

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

    with urlopen(request, timeout=60) as response:
        data = json.load(response)

    versions = data.get("content_versions", [])

    dates = [
        item["date"]
        for item in versions
        if item.get("date")
    ]

    return max(dates)


def download_title(title_number, version_date):

    url = (
        f"https://www.ecfr.gov/api/versioner/v1/full/"
        f"{version_date}/title-{title_number}.xml"
    )

    print(f"\nDownloading Title {title_number}")
    print(f"Version date: {version_date}")

    request = Request(
        url,
        headers={
            "User-Agent": "Legal-Agentic-AI-Research/1.0"
        }
    )

    with urlopen(request, timeout=300) as response:

        data = response.read()

    output_file = (
        OUTPUT_DIR / f"title_{title_number}.xml"
    )

    output_file.write_bytes(data)

    print(
        f"Success! "
        f"{len(data):,} bytes saved."
    )


def main():

    for title_number in FAILED_TITLES:

        for attempt in range(1, 4):

            print(
                f"\nTitle {title_number} "
                f"- Attempt {attempt}/3"
            )

            try:

                version_date = get_latest_version(
                    title_number
                )

                download_title(
                    title_number,
                    version_date
                )

                break

            except Exception as error:

                print(
                    f"Attempt failed: {error}"
                )

                if attempt < 3:

                    print(
                        "Waiting 10 seconds "
                        "before retry..."
                    )

                    time.sleep(10)

                else:

                    print(
                        f"Title {title_number} "
                        f"could not be downloaded."
                    )


if __name__ == "__main__":
    main()