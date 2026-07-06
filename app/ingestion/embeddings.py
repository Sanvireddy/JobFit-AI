import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from app.db.insert_data import (
    get_job_details_by_faiss_indices as db_get_job_details_by_faiss_indices,
    get_job_metadata_by_job_ids as db_get_job_metadata_by_job_ids,
    get_latest_job_ids_with_null_faiss_index,
    update_faiss_index,
)
from app.ingestion.metata_extractor import extract_metadata


ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT_DIR / "job_desc.index"
DIMENSIONS = 384

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_or_create_faiss_index():
    if INDEX_PATH.exists():
        try:
            return faiss.read_index(str(INDEX_PATH))
        except Exception as err:
            print(
                f"Warning: failed to read FAISS index at {INDEX_PATH}. "
                f"Creating a new index instead. Error: {err}"
            )

    return faiss.IndexFlatIP(DIMENSIONS)


def store_embeddings():

    response_list = (get_latest_job_ids_with_null_faiss_index())

    if not response_list:
        print("No new jobs found.")
        return

    # Load FAISS once
    index = load_or_create_faiss_index()

    for response in response_list:
        try:
            job_id, job_description = response
            # Skip empty descriptions
            if not job_description:
                print(f"Skipping job_id={job_id} because description is empty.")
                continue

            # Generate embedding
            embedding = embedding_model.encode([job_description],normalize_embeddings=True)

            vector = np.array(
                embedding,
                dtype=np.float32
            )

            # Add vector to FAISS
            index.add(vector)

            # Get position of newly added vector
            faiss_position = (
                index.ntotal - 1
            )

            # Update DB mapping
            update_faiss_index(
                job_id,
                faiss_position
            )

            print(
                f"Stored embedding for "
                f"job_id={job_id} "
                f"at faiss_position={faiss_position}"
            )

        except Exception as e:

            print(
                f"Failed processing "
                f"job_id={response[0]}"
            )

            print(e)

    # Save FAISS once after batch
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    print("FAISS index saved successfully.")

# index = faiss.read_index(str(INDEX_PATH))
# print(f"Current number of vectors in FAISS index: {index.ntotal}")
# print(f"Dimensionality of vectors: {index.d}")
# first_vector = index.reconstruct(0)
# print(f"First vector (job description embedding): {first_vector}")

def get_resume_embedding(resume_text):
    embedding = embedding_model.encode(
        [resume_text],
        normalize_embeddings=True
    )
    return np.array(embedding, dtype=np.float32)


def search_similar_jobs(resume_embedding, top_k=5):
    """Search for top-k similar jobs based on resume text."""
    
    index = load_or_create_faiss_index()
    if index.ntotal == 0:
        print("FAISS index is empty. No jobs to search.")
        return [], []
    distances, indices = index.search(resume_embedding, top_k)
    return indices[0].tolist(), distances[0].tolist()


def get_job_details_by_faiss_indices(faiss_indices: List[int]) -> List[Dict]:
    """Retrieve full job details from database using FAISS indices."""
    return db_get_job_details_by_faiss_indices(faiss_indices)


def get_job_metadata_by_job_ids(job_ids: List[str]) -> Dict[str, Dict]:
    """Retrieve job metadata from database using job IDs."""
    return db_get_job_metadata_by_job_ids(job_ids)


def filter_jobs_by_metadata(
    jobs: List[Dict],
    job_metadata_map: Dict[str, Dict],
    candidate_experience_years: Optional[int] = None,
) -> List[Dict]:
    """Keep only jobs the candidate is plausibly compatible with.

    A job is kept when all of the following hold:
    - Its required minimum experience is <= the candidate's experience. Jobs
      demanding MORE experience than the candidate has are dropped. (When
      ``candidate_experience_years`` is None, the experience check is skipped.)
    - It does not explicitly require an advanced degree (Master's/PhD).
    - It does not explicitly require a non-English language. Jobs whose
      language requirement is unknown (``requires_only_english`` is None) are
      kept rather than excluded, since "unknown" should not silently filter
      out otherwise-relevant matches.
    """

    filtered_jobs = []

    for job in jobs:
        job_id = job.get("job_id")
        metadata = job_metadata_map.get(job_id, {})

        # Experience: drop jobs requiring more years than the candidate has.
        # Be defensive about None / non-numeric values in the DB.
        raw_exp = metadata.get("min_experience_years")
        try:
            job_min_experience_years = int(raw_exp) if raw_exp is not None else 0
        except (TypeError, ValueError):
            job_min_experience_years = 0

        if (
            candidate_experience_years is not None
            and job_min_experience_years > candidate_experience_years
        ):
            continue

        # Education: drop jobs that explicitly require a Master's/PhD.
        if metadata.get("requires_advanced_degree"):
            continue

        # Language: drop only jobs that explicitly require a non-English
        # language (requires_only_english is explicitly False). Keep jobs that
        # are English-only (True) or have an unknown requirement (None).
        if metadata.get("requires_only_english") is False:
            continue

        filtered_jobs.append(job)

    return filtered_jobs


def find_matching_jobs_for_resume(
    resume_text: str,
    top_k: int = 10,
    apply_metadata_filtering: bool = True,
    candidate_experience_years: Optional[int] = 3,
) -> Dict:
    """
    Complete pipeline: Resume → Metadata Extraction → Filtering →
    Embedding → FAISS Search → Top Matching Jobs

    Args:
        resume_text: The resume content as text
        top_k: Number of top matching jobs to return
        apply_metadata_filtering: Whether to filter jobs by metadata compatibility
        candidate_experience_years: Years of experience the candidate has; jobs
            requiring more than this are filtered out. Pass None to skip the
            experience check. (TODO: derive this from the resume instead of a
            fixed default.)

    Returns:
        Dictionary containing matching jobs and metadata
    """
    result = {
        "success": False,
        "resume_embedding": None,
        "similar_jobs": [],
        "faiss_scores": [],
        "error": None
    }
    
    try:  
        # Step 1: Generate resume embedding
        print("Step 1: Generating resume embedding...")
        resume_embedding = get_resume_embedding(resume_text)
        result["resume_embedding"] = resume_embedding.tolist()
        
        # Step 2: Search FAISS index for similar jobs
        print(f"Step 2: Searching FAISS index for top {top_k} similar jobs...")
        faiss_indices, scores = search_similar_jobs(resume_embedding, top_k=top_k * 2)
        result["faiss_scores"] = scores
        
        if not faiss_indices:
            result["error"] = "No jobs found in FAISS index"
            return result
        
        # Step 3: Retrieve job details
        print("Step 3: Retrieving job details...")
        jobs = get_job_details_by_faiss_indices(faiss_indices)
        job_ids = [job["job_id"] for job in jobs]
        
        # Step 4: Apply metadata filtering
        if apply_metadata_filtering:
            print("Step 4: Filtering jobs by metadata compatibility...")
            job_metadata_map = get_job_metadata_by_job_ids(job_ids)
            jobs = filter_jobs_by_metadata(
                jobs, job_metadata_map, candidate_experience_years
            )
        
        # Return top-k after filtering
        result["similar_jobs"] = jobs[:top_k]
        result["success"] = True
        
        print(f"Found {len(result['similar_jobs'])} matching jobs")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"Error in matching pipeline: {e}")
    
    return result


def display_matching_results(results: Dict) -> None:
    """Pretty print the matching results."""
    if not results["success"]:
        print(f"Matching failed: {results['error']}")
        return
    
    print("\n" + "="*80)
    print("📋 RESUME ANALYSIS")
    print("="*80)
    
    print("\n" + "="*80)
    print("TOP MATCHING JOBS")
    print("="*80)
    
    for idx, job in enumerate(results["similar_jobs"], 1):
        score = results["faiss_scores"][idx-1] if idx-1 < len(results["faiss_scores"]) else 0
        print(f"\n{idx}. {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
        print(f"   Location: {job.get('location', 'N/A')}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   URL: {job.get('application_url', 'N/A')}")


# Example usage of the complete pipeline
if __name__ == "__main__":
    # Sample resume
    sample_resume = """
    Senior Software Engineer with 5+ years of experience in Python, Machine Learning, and MLOps.
    Expertise in:
    - Python, Scala, SQL
    - Machine Learning frameworks: PyTorch, TensorFlow, Scikit-learn
    - MLOps tools: Kubernetes, Docker, Apache Airflow, Databricks
    - Cloud platforms: AWS, GCP
    - NLP and LLM fine-tuning
    
    Education: Master's degree in Computer Science
    Languages: English (native), German (B2)
    
    Currently seeking roles in ML Engineering, MLOps, or Data Science.
    Open to remote positions or relocation with visa sponsorship.
    """
    
    # Run the complete matching pipeline
    results = find_matching_jobs_for_resume(
        resume_text=sample_resume,
        top_k=10,
        apply_metadata_filtering=True
    )
    
    # Display results
    display_matching_results(results)

