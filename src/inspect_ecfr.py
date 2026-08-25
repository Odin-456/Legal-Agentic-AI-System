import xml.etree.ElementTree as ET
from pathlib import Path


file_path = Path("data/raw/ecfr/title_24.xml")

print("Loading eCFR XML...")

tree = ET.parse(file_path)
root = tree.getroot()

print("XML loaded successfully!")

# Find Title 24
title = root.find("DIV1")

# Find Subtitle A
subtitle = title.find("DIV2")

# Find Part 1
part = None

for child in subtitle:
    if child.tag == "DIV5" and child.attrib.get("N") == "1":
        part = child
        break

if part is None:
    print("Part 1 not found.")
    exit()

# Find Section 1.1
section = None

for child in part:
    if child.tag == "DIV8" and child.attrib.get("N") == "1.2":
        section = child
        break

if section is None:
    print("Section 1.1 not found.")
    exit()

print("\nSection 1.1:")
print("Tag:", section.tag)
print("Attributes:", section.attrib)

print("\nChildren of Section 1.1:")

for child in section:
    print("Tag:", child.tag)
    print("Attributes:", child.attrib)

    # Show the text directly inside this element
    text = "".join(child.itertext()).strip()

    if text:
        print("Text:", text[:500])