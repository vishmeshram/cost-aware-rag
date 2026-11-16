from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# =====================================================
# STEP 1 - LOAD MODEL
# =====================================================

print("\n[STEP 1] Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

dim = model.get_sentence_embedding_dimension()

print("Embedding dimension:", dim)

# =====================================================
# STEP 2 - CONNECT TO MILVUS LITE
# =====================================================

print("\n[STEP 2] Connecting to Milvus Lite...")

connections.connect(
    alias="default",
    uri="./demo_milvus.db"
)

print("Connected.")

# =====================================================
# STEP 3 - REFRESH COLLECTION
# =====================================================

collection_name = "sentence_demo"

if utility.has_collection(collection_name):
    print("\nOld collection found.")
    print("Deleting old collection...")

    utility.drop_collection(collection_name)

    print("Collection deleted.")

# =====================================================
# STEP 4 - CREATE COLLECTION
# =====================================================

print("\n[STEP 4] Creating collection...")

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True
    ),

    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=500
    ),

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=dim
    )
]

schema = CollectionSchema(
    fields=fields,
    description="Learning Vector Database"
)

collection = Collection(
    name=collection_name,
    schema=schema
)

print("Collection created.")

# =====================================================
# STEP 5 - DATA
# =====================================================

sentences = [
    "Java needs a lot of efforts in coding",
    "Python is a easy coding language",
    "Interviewer asks code for placement"
]

print("\n[STEP 5] Original Sentences")

for idx, sentence in enumerate(sentences, start=1):
    print(f"{idx}. {sentence}")

# =====================================================
# STEP 6 - CREATE EMBEDDINGS
# =====================================================

print("\n[STEP 6] Encoding sentences...")

embeddings = model.encode(sentences)

for text, vector in zip(sentences, embeddings):

    print("\n--------------------------------")
    print("Sentence:")
    print(text)

    print("\nVector Length:")
    print(len(vector))

    print("\nFirst 10 Dimensions:")
    print(vector[:10])

# =====================================================
# STEP 7 - INSERT INTO DB
# =====================================================

print("\n[STEP 7] Inserting into Milvus...")

ids = [1, 2, 3]

collection.insert(
    [
        ids,
        sentences,
        embeddings.tolist()
    ]
)

collection.flush()

print("Data inserted.")

# =====================================================
# STEP 8 - SHOW STORED DATA
# =====================================================

print("\n[STEP 8] Reading Data From DB")

stored_data = collection.query(
    expr="id >= 0",
    output_fields=["id", "text", "embedding"]
)

for row in stored_data:

    print("\n===================================")
    print("ID:", row["id"])

    print("\nTEXT:")
    print(row["text"])

    print("\nVECTOR LENGTH:")
    print(len(row["embedding"]))

    print("\nFIRST 5 VALUES:")
    print(row["embedding"][:5])

# =====================================================
# STEP 9 - CREATE INDEX
# =====================================================

print("\n[STEP 9] Creating Index...")

index_params = {
    "index_type": "FLAT",
    "metric_type": "COSINE"
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

collection.load()

print("Index created.")

# =====================================================
# STEP 10 - SEARCH QUERY
# =====================================================

query = "which programming language is tough"

print("\n[STEP 10] Query")
print(query)

query_embedding = model.encode([query])

print("\nQuery Vector Length:")
print(len(query_embedding[0]))

print("\nQuery First 10 Values:")
print(query_embedding[0][:10])

# =====================================================
# STEP 11 - SEARCH
# =====================================================

print("\n[STEP 11] Searching...")

results = collection.search(
    data=query_embedding.tolist(),
    anns_field="embedding",
    param={},
    limit=3,
    output_fields=["text"]
)

# =====================================================
# STEP 12 - SHOW RANKING
# =====================================================

print("\n[STEP 12] Similarity Ranking")

for hits in results:

    rank = 1

    for hit in hits:

        print("\n------------------------")
        print("Rank:", rank)

        print("Sentence:")
        print(hit.entity.get("text"))

        print("Similarity Score:")
        print(hit.score)

        rank += 1

# =====================================================
# STEP 13 - BEST MATCH
# =====================================================

best = results[0][0]

print("\n====================================")
print("FINAL RESULT")
print("====================================")

print("Query:")
print(query)

print("\nBest Match:")
print(best.entity.get("text"))

print("\nScore:")
print(best.score)