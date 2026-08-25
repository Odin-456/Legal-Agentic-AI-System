import xml.etree.ElementTree as ET
from pathlib import Path
from langchain_core.documents import Document


INPUT_FILE = Path("data/raw/ecfr/title_24.xml")


def parse_ecfr(file_path):
    print("Loading eCFR XML...")

    tree = ET.parse(file_path)
    root = tree.getroot()

    documents = []

    title = root.find("DIV1")

    if title is None:
        raise ValueError("Title not found.")

    title_number = title.attrib.get("N")

    for subtitle in title:

        if subtitle.tag != "DIV2":
            continue

        subtitle_number = subtitle.attrib.get("N")

        for part in subtitle:

            if part.tag != "DIV5":
                continue

            part_number = part.attrib.get("N")

            for section in part:

                if section.tag != "DIV8":
                    continue

                section_number = section.attrib.get("N")

                heading = ""
                paragraphs = []

                for element in section:

                    if element.tag == "HEAD":
                        heading = "".join(
                            element.itertext()
                        ).strip()

                    elif element.tag == "P":
                        text = "".join(
                            element.itertext()
                        ).strip()

                        if text:
                            paragraphs.append(text)

                if not paragraphs:
                    continue

                content = "\n".join(paragraphs)

                document = Document(
                    page_content=content,
                    metadata={
                        "source": "eCFR",
                        "title": title_number,
                        "subtitle": subtitle_number,
                        "part": part_number,
                        "section": section_number,
                        "heading": heading
                    }
                )

                documents.append(document)

    return documents


if __name__ == "__main__":

    documents = parse_ecfr(INPUT_FILE)

    print("\nParsing complete!")
    print(f"Documents created: {len(documents)}")

    print("\nFirst document:")
    print("-" * 60)

    print("Content:")
    print(documents[0].page_content)

    print("\nMetadata:")
    print(documents[0].metadata)