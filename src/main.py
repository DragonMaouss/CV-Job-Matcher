import kagglehub
import pandas as pd
from preprocess import preprocess_text
from matcher import compute_similarity_matrix


# Download the dataset from Kaggle
path = kagglehub.dataset_download("surendra365/recruitement-dataset")

# Load the dataset into a pandas DataFrame
df = pd.read_csv(f"{path}/job_applicant_dataset.csv")

# Extract the "Resume" and "Job_Description" columns
resumes_df = df.loc[:, "Resume"]
jobs_df = df.loc[:, "Job Description"]

# Save the extracted columns as separate CSV files
resumes_df.to_csv("../data/resumes.csv", index=False)
jobs_df.to_csv("../data/job.csv", index=False)

# Preprocess all texts
clean_resumes = [preprocess_text(r) for r in resumes_df]
clean_jobs = [preprocess_text(j) for j in jobs_df]

similarity_matrix = compute_similarity_matrix(clean_resumes, clean_jobs)

# Matching
results = []
for i in range(len(clean_resumes)):
    # Find the best matching job for each resume
    best_job_idx = similarity_matrix[i].argmax()
    best_score = similarity_matrix[i][best_job_idx]
    results.append(
        {"cv_id": i, "job_id": int(best_job_idx), "score": float(best_score)}
    )

# Sort by best matches and save
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="score", ascending=False)
results_df.to_csv("../data/matching_results.csv", index=False)

print(results_df.head(10))
