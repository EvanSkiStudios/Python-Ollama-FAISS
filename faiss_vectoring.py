import faiss
import numpy as np
import os
import json
import ollama


def create_vector_db_from_json(json_file_name):
    # Load JSON file
    running_dir = os.path.dirname(os.path.realpath(__file__))
    files_location = str(running_dir) + "\\dir\\"
    testing_json = os.path.join(files_location, json_file_name)

    # File paths
    index_path = os.path.join(files_location, json_file_name + "faiss_index.bin")
    metadata_path = os.path.join(files_location, json_file_name + "metadata.json")
    embeddings_path = os.path.join(files_location, json_file_name + "embeddings.npy")

    with open(testing_json, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Role + content for embedding meaning
    embedding_texts = [
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    ]

    # Metadata stays separate
    metadata = [
        {
            "role": msg["role"],
            "name": msg.get("name"),
            "content": msg["content"]
        }
        for msg in messages
    ]

    # generate embeddings
    embeddings = []
    for text in embedding_texts:
        resp = ollama.embed(
            model="snowflake-arctic-embed2:latest",
            input=text
        )
        # print(resp)
        embeddings.append(resp["embeddings"][0])

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    # apply embeddings to database
    index = faiss.IndexFlatL2(dim)
    # noinspection PyArgumentList
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save embeddings (optional)
    np.save(embeddings_path, embeddings)

    return index, metadata, embeddings


def search(query, index, metadata, k_results=3, max_distance=10.0):
    # Embed the query
    resp = ollama.embed(
        model="snowflake-arctic-embed2:latest",
        input=query
    )

    # Extract the actual vector
    q_embed_vector = resp["embeddings"][0]

    # Convert to numpy array for FAISS
    q_embed = np.array([q_embed_vector], dtype="float32")

    # Cap k_results to the actual number of vectors in the index
    k_results = min(k_results, index.ntotal)

    # Search
    distances, indices = index.search(q_embed, k_results)

    # Return metadata results
    results = []
    # Loop through the retrieved indices and distances, keeping track of ranking order
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        # Skip results that exceed the maximum allowed distance (i.e., too dissimilar)
        if dist > max_distance:
            continue
        # Append a structured dictionary with the relevant metadata and ranking info
        results.append({
            "rank": rank + 1,  # Rank starts at 1, not 0
            "distance": float(dist),  # Convert distance to float for consistency
            "role": metadata[idx]["role"],  # Role information from metadata
            "name": metadata[idx]["name"],  # Name associated with the entry
            "content": metadata[idx]["content"]  # The actual content or text
        })

    return results


def testing(index, metadata):
    query_text = "why is the sky blue?"
    matches = search(query_text, index, metadata, 20)

    for match in matches:
        print(match)
        #print(f"[{match['rank']}] ({match['role']}) {match['name'] or ''} -> {match['content']} [dist={match['distance']:.4f}]")


def main():
    args = create_vector_db_from_json('testing.json')
    print(args[0])

    testing(args[0], args[1])


if __name__ == "__main__":
    main()
