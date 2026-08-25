from pathlib import Path
file_path = Path("data/raw/sample_law.txt")
text = file_path.read_text(encoding="utf_8")
print(text)

print("LEGAL DOCUMENT")
print("=" * 50)
print(text)