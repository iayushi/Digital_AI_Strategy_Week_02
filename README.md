# Digital AI Strategy - Week 1

## Course Overview

Welcome to the **Digital AI Strategy** course! This repository contains materials and resources for Week 1, focusing on the **Foundations of IS, IT, Digital Platform, AI, and Business Strategy**.

### What's Included

- **DAIS_Week1_RAG.py**: Interactive Streamlit application with RAG (Retrieval Augmented Generation) capabilities
- **Course Materials**: Vector database and supporting resources

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run DAIS_Week1_RAG.py`
3. Enter your API key for your preferred LLM provider
4. Start asking questions about Digital AI Strategy concepts!

## 🗄️ Vector Database Implementation

### Introduction to Vector Databases

Vector databases are specialized storage systems designed to handle high-dimensional vector embeddings efficiently. In the context of this Digital AI Strategy course, vector databases serve several critical purposes:

- **Semantic Search**: Enable finding conceptually similar content rather than just keyword matches
- **Knowledge Retrieval**: Power the RAG (Retrieval Augmented Generation) system to provide contextually relevant answers
- **Course Material Organization**: Store and index course content for intelligent querying and exploration
- **Scalable Information Access**: Handle large volumes of educational content efficiently

The implementation uses **ChromaDB**, a popular open-source vector database, combined with **HuggingFace embeddings** to create a sophisticated retrieval system for course materials.

### How Vector Embeddings Work

Vector embeddings transform text content into numerical representations (vectors) that capture semantic meaning. Similar concepts end up close together in vector space, enabling:

1. **Semantic similarity matching** - Find related concepts even with different wording
2. **Context-aware retrieval** - Understand the meaning behind queries
3. **Efficient similarity search** - Quickly find relevant course materials

### Current Implementation Overview

The application uses the following vector database setup:

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers)
- **Vector Store**: ChromaDB with local persistence
- **Storage Location**: `./Week_1_31Aug2025/` directory
- **Retrieval Method**: Similarity search with top-k results (k=5)

### Step-by-Step Guide: Generating Vector Embeddings

#### 1. Prepare Your Course Content

```python
# Example: Prepare documents for embedding
from langchain.schema import Document

# Your course materials as text chunks
course_materials = [
    "Information Systems form the backbone of digital operations...",
    "Digital Platforms create ecosystems for digital interactions...",
    "Artificial Intelligence enhances decision-making capabilities...",
    # Add more content
]

# Convert to Document objects
documents = [
    Document(page_content=content, metadata={"source": f"chunk_{i}"})
    for i, content in enumerate(course_materials)
]
```

#### 2. Initialize the Embedding Model

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the same embedding model used in the application
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 3. Create and Populate the Vector Database

```python
from langchain_community.vectorstores import Chroma

# Create new vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./your_course_vectordb"
)

# Persist the database to disk
vectorstore.persist()
```

#### 4. Test the Vector Database

```python
# Test similarity search
query = "What are digital platforms?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content[:100]}...")
```

### Setting Up and Configuring the Vector Database

#### Option 1: Using the Pre-built Database

The repository includes a pre-built vector database in the `Week_1_31Aug2025` directory. To use it:

1. Ensure the directory exists and contains the ChromaDB files
2. Update the `PERSIST_DIRECTORY` variable in `DAIS_Week1_RAG.py`:
   ```python
   PERSIST_DIRECTORY = "./Week_1_31Aug2025"
   ```
3. Run the application - it will automatically load the existing database

#### Option 2: Creating Your Own Database

1. **Prepare your content**: Organize course materials into text chunks
2. **Install required packages**: Ensure you have all dependencies from `requirements.txt`
3. **Create the database**:

```python
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3  # Required for ChromaDB compatibility

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Your implementation here
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load your documents
documents = [...]  # Your course content as Document objects

# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./your_custom_vectordb"
)
```

4. **Update the application**: Modify `PERSIST_DIRECTORY` to point to your new database

### Configuration Options and Customization

#### Embedding Model Options

You can customize the embedding model based on your needs:

```python
# Option 1: Multilingual support
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Option 2: Higher accuracy (larger model)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Option 3: Domain-specific models
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/allenai-specter"  # Scientific papers
)
```

#### Search Configuration

Customize the retrieval behavior:

```python
# In the run_similarity_search function
def run_similarity_search(query):
    # Adjust k for more/fewer results
    results = vectorstore.similarity_search(
        query, 
        k=10,  # Return top 10 instead of 5
        filter={"source": "specific_topic"}  # Optional filtering
    )
    return results

# Alternative: Use similarity search with score threshold
def run_similarity_search_with_score(query):
    results = vectorstore.similarity_search_with_score(
        query, 
        k=5,
        score_threshold=0.7  # Only return results above similarity threshold
    )
    return [doc for doc, score in results if score > 0.7]
```

#### Database Configuration

Customize ChromaDB settings:

```python
# Advanced ChromaDB configuration
vectorstore = Chroma(
    persist_directory="./custom_vectordb",
    embedding_function=embedding_model,
    collection_name="course_materials",  # Custom collection name
    collection_metadata={"description": "Digital AI Strategy Course Content"}
)
```

### Examples of Extending Functionality

#### 1. Adding New Course Materials

```python
# Add new documents to existing database
new_documents = [
    Document(page_content="New course content...", metadata={"week": "2"})
]

vectorstore.add_documents(new_documents)
vectorstore.persist()
```

#### 2. Multi-Collection Setup

```python
# Create separate collections for different course weeks
week1_vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_name="week1_materials"
)

week2_vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_name="week2_materials"
)
```

#### 3. Metadata Filtering

```python
# Search with metadata filters
def search_by_topic(query, topic):
    results = vectorstore.similarity_search(
        query,
        k=5,
        filter={"topic": topic}
    )
    return results

# Usage
ai_results = search_by_topic("machine learning", "AI")
strategy_results = search_by_topic("planning", "Business Strategy")
```

#### 4. Hybrid Search (Vector + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine vector similarity with keyword search
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
keyword_retriever = BM25Retriever.from_documents(documents)

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.7, 0.3]  # 70% vector, 30% keyword
)
```

#### 5. Custom Similarity Metrics

```python
# Use different distance metrics
vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  # Options: l2, ip, cosine
)
```

### Performance Optimization Tips

1. **Batch Processing**: Process documents in batches for better performance
2. **GPU Acceleration**: Use CUDA for embedding generation if available
3. **Chunking Strategy**: Optimize text chunk size (typically 200-1000 tokens)
4. **Caching**: Cache embeddings to avoid recomputation
5. **Index Optimization**: Tune ChromaDB parameters for your use case

### Troubleshooting Common Issues

#### Issue 1: SQLite3 Compatibility
```python
# Fix for ChromaDB SQLite issues
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
```

#### Issue 2: CUDA/GPU Issues
```python
# Force CPU usage if GPU issues
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

#### Issue 3: Memory Issues with Large Datasets
```python
# Process in smaller batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
    vectorstore.persist()
```

### 🔧 Helper Script: create_embeddings.py

The repository includes a powerful helper script that simplifies creating vector embeddings from your course content. This script allows you to easily convert .md or .txt files into vector embeddings and store them in a ChromaDB database.

#### Features

- **Multiple File Support**: Process multiple .md or .txt files at once
- **Customizable Chunking**: Adjust chunk sizes and overlap for optimal performance
- **GPU/CPU Support**: Automatically detect and use GPU acceleration when available
- **Database Management**: Create new databases or update existing ones
- **Error Handling**: Comprehensive error handling with user-friendly feedback
- **Testing**: Built-in database testing with sample queries

#### Basic Usage

```bash
# Create embeddings from a single file
python create_embeddings.py course_material.md

# Process multiple files
python create_embeddings.py file1.md file2.txt file3.md

# Process all markdown files in a directory
python create_embeddings.py *.md
```

#### Advanced Usage

```bash
# Use custom chunk size and enable GPU
python create_embeddings.py --chunk-size 1000 --use-gpu files/*.md

# Specify custom database location
python create_embeddings.py --db-path ./my_vectordb files/*.txt

# Use a different embedding model
python create_embeddings.py --model sentence-transformers/all-mpnet-base-v2 *.md

# Create with custom collection name
python create_embeddings.py --collection-name "week2_materials" week2/*.md
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name for embeddings | `all-MiniLM-L6-v2` |
| `--chunk-size` | Size of text chunks for processing | `500` |
| `--chunk-overlap` | Overlap between text chunks | `50` |
| `--use-gpu` | Use GPU for embedding generation if available | `false` |
| `--db-path` | Directory to store the vector database | `./course_vectordb` |
| `--collection-name` | Name of the vector database collection | `course_materials` |
| `--test-query` | Query to test the database with | `"digital strategy"` |
| `--no-test` | Skip testing the vector database after creation | `false` |

#### Integration with DAIS_Week1_RAG.py

After creating your vector database with the helper script, update the `PERSIST_DIRECTORY` variable in `DAIS_Week1_RAG.py`:

```python
# Update this line in DAIS_Week1_RAG.py
PERSIST_DIRECTORY = "./your_custom_vectordb"  # Path from --db-path option
```

#### Example Workflow

1. **Prepare your content**: Organize course materials into .md or .txt files
2. **Create embeddings**: Use the helper script to process your files
3. **Update the app**: Point the Streamlit app to your new database
4. **Test**: Run the app and verify that your content is searchable

```bash
# Step 1: Create embeddings from your course materials
python create_embeddings.py --chunk-size 800 --use-gpu course_materials/*.md

# Step 2: Update DAIS_Week1_RAG.py to use your database
# PERSIST_DIRECTORY = "./course_vectordb"

# Step 3: Run the Streamlit app
streamlit run DAIS_Week1_RAG.py
```

#### Troubleshooting the Helper Script

**Issue 1: Model Download Errors**
```bash
# Ensure you have internet connectivity for first-time model download
# Models are cached locally after first download
```

**Issue 2: Memory Issues with Large Files**
```bash
# Use smaller chunk sizes for large files
python create_embeddings.py --chunk-size 300 large_file.md
```

**Issue 3: GPU Issues**
```bash
# Force CPU usage if GPU issues occur
python create_embeddings.py large_file.md  # GPU detection is automatic
```

### Best Practices for Course Material Implementation

1. **Content Preparation**:
   - Clean and preprocess text content
   - Maintain consistent formatting
   - Include relevant metadata (topics, weeks, difficulty level)

2. **Chunking Strategy**:
   - Split content into logical, coherent chunks
   - Overlap chunks slightly to maintain context
   - Size chunks appropriately (aim for 200-500 tokens)

3. **Database Management**:
   - Regular backups of vector databases
   - Version control for database schemas
   - Monitor database size and performance

4. **Quality Assurance**:
   - Test retrieval quality with sample queries
   - Validate that similar concepts are found together
   - Monitor and improve relevance over time

## Sample Questions to Explore Course Concepts

This collection of sample questions is designed to help you chat with this bot to understand Digital AI Strategy session content through relatable and creative analogies.

## 🦸‍♂️ Pop Culture & Marvel Universe

### Marvel-Inspired Questions

**"How are content in this session related to Digital Platform, AI, and Business Strategy ? Explain to a Marvel fan."**
**"If the Digital Transformation was an Avenger, which one would it be and why?"**


**"If IS Strategy making team would be a superhero team, what would be its superpowers?"**

## 👶 Child-Friendly Explanations

### Simple Language Questions

**"Explain the learnings from this course to a five year old"**
- Use everyday objects and activities
- Focus on "helping" and "making things better"
- Think about toys, games, and familiar experiences
- Break down cause-and-effect relationships

**"How would you explain AI to someone who has never used a computer?"**
- Compare to helpful assistants or smart helpers
- Use analogies to human learning and memory
- Focus on problem-solving capabilities

## 🎮 Gaming & Interactive Analogies

### Gaming-Inspired Questions

**"How would you explain Digital and AI strategy content in this session to someone who loves video games?"**
- Gaming consoles as platforms (PlayStation, Xbox, Nintendo)
- Game ecosystems with developers, players, and content
- App stores and digital marketplaces
- Multiplayer networks and communities
- Platform rules and developer tools

**"Compare building an AI system to creating a video game character"**
- Training = leveling up and gaining skills
- Data = experience points and learning from gameplay
- Algorithms = character abilities and behaviors
- Testing = beta testing and balancing

## 🧱 Building & Construction Analogies

### LEGO & Construction Questions

**"Compare AI Strategy to building a LEGO masterpiece"**
- Planning: Having the right blueprint and vision
- Components: Different LEGO pieces = different AI technologies
- Foundation: Basic blocks that everything else builds upon
- Integration: How pieces connect and work together
- Iteration: Building, testing, and improving the design

**"How is implementing AI in business like renovating a house?"**
- Assessment: Understanding current capabilities and needs
- Planning: Designing the renovation approach
- Infrastructure: Upgrading foundational systems
- Integration: Making new and old systems work together
- Timeline: Phased implementation and testing

## 🏙️ Ecosystem & Community Analogies

### City Building Questions

**"Compare developing a Digital Platform to creating a city"**
- Infrastructure: Roads, utilities, and basic services
- Zoning: Different areas for different purposes
- Residents: Users who live and work in the ecosystem
- Businesses: Third-party services and applications
- Government: Platform rules and governance
- Growth: Expansion and sustainable development

**"How is Business Strategy like coaching a sports team?"**
- Game plan: Overall strategy and tactics
- Player development: Building capabilities and skills
- Resource allocation: Putting the right players in the right positions
- Adaptation: Adjusting strategy based on opponents and conditions
- Performance measurement: Tracking progress and results

## 🎵 Creative & Artistic Analogies

### Orchestra & Music Questions

**"How do IS, Digital Platform, AI, and Business Strategy work together like instruments in an orchestra?"**
- Conductor: Business Strategy directing the overall performance
- String section: Information Systems providing the foundation
- Brass section: AI providing powerful, attention-grabbing capabilities
- Woodwinds: Digital Platforms creating harmony and connection
- Percussion: Implementation that drives the rhythm of change

**"Compare AI development to composing a symphony"**
- Composition: Designing the AI architecture
- Musicians: Different algorithms and models
- Practice: Training and optimization
- Performance: Deployment and real-world application
- Audience: End users and stakeholders

## 🔍 Advanced Integration Questions

### Cross-Concept Connections

**"If you were building a digital company from scratch, how would you use concepts from this sessions topic?"**


**"How do the four main course topics in this session relate to  'digital transformation '?"**

**"Compare the evolution of digital business to the growth of a tree"**


### Future-Thinking Questions

**"If AI could be any kitchen appliance, what would it be and why?"**

**"How is Digital Strategy like planning a space mission?"**

## 🎯 Application-Focused Questions

### Real-World Application

**"How would you explain the value of Digital Platforms to a traditional brick-and-mortar store owner?"**

**"If your favorite hobby became a digital business, how would you apply course concepts?"**


## How to Use These Questions

1. **Choose your favorite analogies**: Start with contexts you're most familiar with
2. **Layer the complexity**: Begin with simple comparisons and add details
3. **Mix and match**: Combine different analogies for richer understanding
4. **Create your own**: Use these as inspiration for personal analogies
5. **Test with the RAG system**: Ask these questions in the course application
6. **Discuss with peers**: Share analogies and learn from others' perspectives

## Tips for Creating Your Own Analogies

- **Start with what you know**: Use your hobbies, interests, and experiences
- **Focus on relationships**: How do components interact and depend on each other?
- **Consider scale**: How do systems grow and evolve over time?
- **Think about problems**: What challenges exist and how are they solved?
- **Explore benefits**: What value is created and for whom?

Remember: The goal is to make complex concepts accessible and memorable. The best analogies are the ones that resonate with your personal experience and help you see familiar patterns in new contexts!*
