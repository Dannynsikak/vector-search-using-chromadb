## ChromaDB with Performance Logic - Python

## Introduction

This project integrates ChromaDB, a powerful vector database, with custom performance optimization logic. It efficiently handles large-scale vector similarity searches, making it ideal for applications such as recommendation engines, content-based retrieval, and AI-powered search systems.

## Key Features:

ChromaDB for fast and scalable vector search.
Custom performance logic for improved query efficiency.
Easy-to-use Python interface for seamless integration.
Table of Contents
Installation
Usage
Performance Considerations
Examples
Contributing
License
Installation
To set up the project, you will need Python 3.8+ and some dependencies.

1. Clone the Repository
   git clone

git@github.com:Dannynsikak/vector-search-using-chromadb.git

cd chromadb-performance 2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate` 3. Install Dependencies

Install the required Python libraries using the following command:

pip install -r requirements.txt

1. Import and Initialize ChromaDB
   The following example demonstrates how to initialize the ChromaDB connection and store vectors:
   from chromadb import ChromaDB

# Initialize ChromaDB with custom performance configurations

db = ChromaDB()
db.initialize()

# Add your data (example: embedding vectors)

embeddings = [...] # A list of embeddings (numpy arrays or lists)
db.add_vectors(embeddings) 2. Perform Vector Search

query_vector = [...] # The vector you want to search
results = db.search(query_vector, top_k=5) # Retrieve top 5 results

for result in results:
print(f"ID: {result['id']}, Score: {result['score']}") 3. Handling Large Datasets
If you're dealing with large datasets, you can implement batching and parallelization for performance improvement.

db.batch_add_vectors(large_embeddings, batch_size=1000) # Add in batches
Performance Considerations
Optimizations 🛠
Embedding Normalization: Normalizing the embeddings helped in faster and more accurate similarity searches. This optimization reduced false positives and improved the overall retrieval speed.

Query Time Reduction: Query execution times were optimized by enhancing the vector search algorithms used in ChromaDB (see the search_query function for reference). This resulted in an X% reduction in query time.

Accuracy Metrics 📊
Cosine Similarity: Cosine similarity is used to measure the accuracy of text-image matching. It is calculated in the calculate_similarity function, which compares the query embedding against stored embeddings, ensuring that higher similarity scores indicate better matches.

Embedding Normalization: Normalization of the embeddings further improved the accuracy of vector searches. This resulted in better matching performance, as demonstrated in the process_images and process_texts functions.

Memory Usage
Monitor memory usage, especially with large vector sets. You may need to use techniques like sharding or external storage solutions for very large datasets.
Example Performance Tuning

from concurrent.futures import ThreadPoolExecutor

def batch_insert(embeddings):
db.batch_add_vectors(embeddings)

with ThreadPoolExecutor(max_workers=4) as executor:
executor.map(batch_insert, split_batches(embeddings))
This code splits the embeddings into batches and inserts them concurrently using multiple threads.

Choosing the Right Distance Metric
Depending on your data, you can choose different distance metrics, such as cosine similarity or Euclidean distance. Here’s an example of
using cosine similarity:

db.set_distance_metric('cosine')

## Full Example

import numpy as np
from chromadb import ChromaDB

# Initialize ChromaDB

db = ChromaDB()
db.initialize()

# Example embeddings (2D numpy array)

embeddings = np.random.random((100, 128))

# Add vectors

db.add_vectors(embeddings)

# Query

query_vector = np.random.random((1, 128))
results = db.search(query_vector, top_k=3)

print("Top 3 results:", results)
Large Dataset Example with Batching
python

large_embeddings = np.random.random((100000, 128)) # Large dataset

# Add embeddings in batches

db.batch_add_vectors(large_embeddings, batch_size=5000)

## Contributing

We welcome contributions! If you'd like to improve performance or add new features to this project, feel free to fork the repository and submit a pull request.

## Development Setup

Fork the repository.
Create a new branch for your feature or fix.
Commit your changes with clear descriptions.
Push the branch to your fork and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
