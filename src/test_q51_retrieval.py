from structure_retrieval import StructureAwareRetriever


question = (
    "Does this provision identify any additional management bodies "
    "or other people who may assist an association?"
)

retriever = StructureAwareRetriever(candidate_k=50)

results = retriever.retrieve(question)

for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(result)
