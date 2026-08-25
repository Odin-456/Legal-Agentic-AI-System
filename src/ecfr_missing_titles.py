import requests


def get_structure(title, date):
    url = (
        f"https://www.ecfr.gov/api/versioner/v1/"
        f"structure/{date}/title-{title}.json"
    )

    print(f"\nGetting structure for Title {title}...")
    print(f"URL: {url}")

    response = requests.get(url, timeout=60)

    print(f"Status: {response.status_code}")

    response.raise_for_status()

    return response.json()


def show_parts(node, level=0):
    """
    Recursively find and display PART nodes.
    """

    if isinstance(node, dict):

        node_type = node.get("type", "").upper()

        if node_type == "PART":
            identifier = node.get("identifier", "")
            label = node.get("label", "")

            print(
                f"Part {identifier}: {label}"
            )

        for value in node.values():
            if isinstance(value, (dict, list)):
                show_parts(value, level + 1)

    elif isinstance(node, list):

        for item in node:
            show_parts(item, level)


def main():

    titles = {
        7: "2025-12-01",
        40: "2026-04-20",
    }

    for title, date in titles.items():

        try:

            data = get_structure(title, date)

            print(f"\nParts found in Title {title}:")
            show_parts(data)

        except Exception as error:

            print(
                f"Error processing Title {title}: {error}"
            )


if __name__ == "__main__":
    main()