import faiss
import numpy as np
import os
import json
import ollama


def build_or_load_faiss_index(json_file_name):
    # Load JSON file
    running_dir = os.path.dirname(os.path.realpath(__file__))
    json_dir = os.path.join(running_dir, "dir")
    testing_json = os.path.join(json_dir, json_file_name)

    # File paths (remove .json from base name)
    base_filename = json_file_name.replace('.json', '')
    index_path = os.path.join(json_dir, base_filename + "faiss_index.bin")
    metadata_path = os.path.join(json_dir, base_filename + "metadata.json")
    embeddings_path = os.path.join(json_dir, base_filename + "embeddings.npy")

    # Load original messages from JSON
    with open(testing_json, "r", encoding="utf-8") as f:
        raw_messages = json.load(f)

    # ------------------------
    # Pair user+assistant messages
    # ------------------------
    paired_messages = []
    i = 0
    while i < len(raw_messages):
        msg = raw_messages[i]

        if msg["role"] == "user" and i + 1 < len(raw_messages) and raw_messages[i + 1]["role"] == "assistant":
            pair = {
                "user": {
                    "role": msg["role"],
                    "name": msg.get("name"),
                    "content": msg["content"]
                },
                "assistant": {
                    "role": raw_messages[i + 1]["role"],
                    "name": raw_messages[i + 1].get("name"),
                    "content": raw_messages[i + 1]["content"]
                }
            }
            paired_messages.append(pair)
            i += 2
        else:
            pair = {
                msg["role"]: {
                    "role": msg["role"],
                    "name": msg.get("name"),
                    "content": msg["content"]
                }
            }
            paired_messages.append(pair)
            i += 1

    # ------------------------
    # Decide whether to regenerate or load cache
    # ------------------------
    needs_regeneration = False

    if not (os.path.exists(metadata_path) and os.path.exists(embeddings_path) and os.path.exists(index_path)):
        needs_regeneration = True
    else:
        # Check cached metadata length matches new one
        with open(metadata_path, "r", encoding="utf-8") as f:
            cached_metadata = json.load(f)

        if len(cached_metadata) != len(paired_messages):
            needs_regeneration = True

    if needs_regeneration:
        print("Regenerating embeddings and FAISS index...")

        # ------------------------
        # Build embedding texts with context
        # ------------------------
        texts_for_embedding = []
        for pair in paired_messages:
            if "user" in pair and "assistant" in pair:
                text = f"USER: {pair['user']['content']} ASSISTANT: {pair['assistant']['content']}"
            elif "user" in pair:
                text = f"USER: {pair['user']['content']}"
            else:
                text = f"ASSISTANT: {pair['assistant']['content']}"
            texts_for_embedding.append(text)

        # ------------------------
        # Generate embeddings
        # ------------------------
        embedding_vectors = []
        for text in texts_for_embedding:
            resp = ollama.embed(
                model="snowflake-arctic-embed2",
                input=text
            )
            embedding_vectors.append(resp["embeddings"][0])

        embedding_vectors = np.array(embedding_vectors, dtype="float32")
        dim = embedding_vectors.shape[1]

        # Build FAISS index
        index = faiss.IndexFlatL2(dim)
        # noinspection PyArgumentList
        index.add(embedding_vectors)

        # Save files
        faiss.write_index(index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(paired_messages, f, ensure_ascii=False, indent=2)
        np.save(embeddings_path, embedding_vectors)

    else:
        print("Loading cached embeddings and FAISS index...")

        # Load cache
        with open(metadata_path, "r", encoding="utf-8") as f:
            paired_metadata = json.load(f)
        embedding_vectors = np.load(embeddings_path)
        index = faiss.read_index(index_path)

    return index, paired_metadata, embedding_vectors


def search_faiss_index(query, index, metadata, top_k=3, max_distance=2.0):
    # Embed the query
    resp = ollama.embed(
        model="snowflake-arctic-embed2",
        input=query
    )

    # Extract the actual vector
    query_vector = resp["embeddings"][0]

    # Convert to numpy array for FAISS
    query_vector_array = np.array([query_vector], dtype="float32")

    # Cap top_k to the actual number of vectors in the index
    top_k = min(top_k, index.ntotal)

    # Search
    distances, indices = index.search(query_vector_array, top_k)

    # Return metadata results
    retrieved_messages = []
    # Loop through the retrieved indices and distances, keeping track of ranking order
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        # Skip results that exceed the maximum allowed distance (i.e., too dissimilar)
        if dist > max_distance:
            continue
        # Append a structured dictionary with the relevant metadata and ranking info
        retrieved_messages.append({
            "rank": rank + 1,  # Rank starts at 1, not 0
            "distance": float(dist),  # Convert distance to float for consistency
            "context": metadata[idx]
        })

    return retrieved_messages


def get_relevant_messages(query, index, metadata):
    query_text = query
    retrieved_matches = search_faiss_index(query_text, index, metadata, 20)

    # print(f"[{match['rank']}] [dist={match['distance']:.4f}] {match['context']} ")

    message_groups = []
    for match in retrieved_matches:
        # Extract messages as a list of dicts
        message_list = list(match["context"].values())
        # Append the pair/group as-is to relevant_memories
        message_groups.append(message_list)

        # Reverse the order of the message groups, not the dicts inside
    message_groups.reverse()

    # Flatten the list of lists into a single list of dicts
    flattened_messages = [msg for group in message_groups for msg in group]

    # Convert the whole flattened list to JSON once
    json_str = json.dumps(flattened_messages, ensure_ascii=False, indent=2)
    return json_str


def main():
    args = build_or_load_faiss_index('evanski_.json')
    print("Index built:", args[0].ntotal, "vectors")

    query = "what is 2+2?"
    memories = get_relevant_messages(query, args[0], args[1])

    print(memories)


if __name__ == "__main__":
    main()
