from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from property_law_parser import parse_property_law_corpus

sections = parse_property_law_corpus()

seen = set()
output = []

for item in sections:
    key = (item["act"], item["section"])

    if key not in seen:
        seen.add(key)
        output.append(
            f'{item["act"]} | Section {item["section"]} | {item["heading"]}'
        )

output.append(f"\nTotal unique sections: {len(seen)}")

output_file = PROJECT_ROOT / "available_sections.txt"
output_file.write_text("\n".join(output), encoding="utf-8")

print(f"Saved {len(seen)} sections to: {output_file}")