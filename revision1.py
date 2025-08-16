import faiss
import numpy as np
import os
import json
import ollama


def create_vector_db_from_json(json_file_name):
    # Load JSON file
    running_dir = os.path.dirname(os.path.realpath(__file__))
    files_location = os.path.join(running_dir, "dir")
    testing_json = os.path.join(files_location, json_file_name)

    # File paths (remove .json from base name)
    memories_file = json_file_name.replace('.json', '')
    index_path = os.path.join(files_location, memories_file + "faiss_index.bin")
    metadata_path = os.path.join(files_location, memories_file + "metadata.json")
    embeddings_path = os.path.join(files_location, memories_file + "embeddings.npy")

    # Load original messages from JSON
    with open(testing_json, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # ------------------------
    # Pair user+assistant messages
    # ------------------------
    paired_metadata = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            pair = {
                "user": {
                    "role": msg["role"],
                    "name": msg.get("name"),
                    "content": msg["content"]
                },
                "assistant": {
                    "role": messages[i + 1]["role"],
                    "name": messages[i + 1].get("name"),
                    "content": messages[i + 1]["content"]
                }
            }
            paired_metadata.append(pair)
            i += 2
        else:
            pair = {
                msg["role"]: {
                    "role": msg["role"],
                    "name": msg.get("name"),
                    "content": msg["content"]
                }
            }
            paired_metadata.append(pair)
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

        if len(cached_metadata) != len(paired_metadata):
            needs_regeneration = True

    if needs_regeneration:
        print("Regenerating embeddings and FAISS index...")

        # ------------------------
        # Build embedding texts with context
        # ------------------------
        embedding_texts = []
        for pair in paired_metadata:
            if "user" in pair and "assistant" in pair:
                text = f"USER: {pair['user']['content']} ASSISTANT: {pair['assistant']['content']}"
            elif "user" in pair:
                text = f"USER: {pair['user']['content']}"
            else:
                text = f"ASSISTANT: {pair['assistant']['content']}"
            embedding_texts.append(text)

        # ------------------------
        # Generate embeddings
        # ------------------------
        embeddings = []
        for text in embedding_texts:
            resp = ollama.embed(
                model="snowflake-arctic-embed2",
                input=text
            )
            embeddings.append(resp["embeddings"][0])

        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]

        # Build FAISS index
        index = faiss.IndexFlatL2(dim)
        # noinspection PyArgumentList
        index.add(embeddings)

        # Save files
        faiss.write_index(index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(paired_metadata, f, ensure_ascii=False, indent=2)
        np.save(embeddings_path, embeddings)

    else:
        print("Loading cached embeddings and FAISS index...")

        # Load cache
        with open(metadata_path, "r", encoding="utf-8") as f:
            paired_metadata = json.load(f)
        embeddings = np.load(embeddings_path)
        index = faiss.read_index(index_path)

    return index, paired_metadata, embeddings


def query_faiss_db(query, index, metadata, k_results=3, max_distance=2.0):
    # Embed the query
    resp = ollama.embed(
        model="snowflake-arctic-embed2",
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
            "context": metadata[idx]
        })

    return results


def gather_relevant_memories_from_database(query, index, metadata):
    query_text = query
    matches = query_faiss_db(query_text, index, metadata, 20)

    # print(f"[{match['rank']}] [dist={match['distance']:.4f}] {match['context']} ")

    relevant_memories = []
    for match in matches:
        # Extract messages as a list of dicts
        messages = list(match["context"].values())
        # Append the pair/group as-is to relevant_memories
        relevant_memories.append(messages)

        # Reverse the order of the message groups, not the dicts inside
    relevant_memories.reverse()

    # Flatten the list of lists into a single list of dicts
    flattened_memories = [msg for group in relevant_memories for msg in group]

    # Convert the whole flattened list to JSON once
    json_str = json.dumps(flattened_memories, ensure_ascii=False, indent=2)
    return json_str


def main():
    args = create_vector_db_from_json('evanski_.json')
    print("Index built:", args[0].ntotal, "vectors")

    query = "what is 2+2?"
    memories = gather_relevant_memories_from_database(query, args[0], args[1])

    print(memories)


if __name__ == "__main__":
    main()
