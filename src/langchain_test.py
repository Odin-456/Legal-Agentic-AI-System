from langchain_core.documents import Document


document = Document(
    page_content="The purpose of this part 1 is to effectuate the provisions of title VI of the Civil Rights Act of 1964.",
    metadata={
        "title": "24",
        "subtitle": "A",
        "part": "1",
        "section": "1.1",
        "source": "eCFR"
    }
)

print("LangChain Document created successfully!")
print()
print("Content:")
print(document.page_content)
print()
print("Metadata:")
print(document.metadata)