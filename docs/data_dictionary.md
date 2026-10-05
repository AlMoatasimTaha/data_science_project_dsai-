# Data dictionary and semantics

The publisher's partial dictionary is retained in `data/raw/data_dictionary.csv`. `reports/data_dictionary.csv` profiles **every column across all five source CSVs**, including dtype, missingness, cardinality and examples.

| Table | Field | Meaning and modeling role |
|---|---|---|
| ai_jobs | job_id | Unique posting key; relationship and partition integrity only |
| ai_jobs | job_title | Original title; six categories; sole available NLP text |
| ai_jobs | company_type | Startup, MNC or Research Lab; organization category, not employer ID |
| ai_jobs | industry | Sector of posting |
| ai_jobs | country | Posting country, six categories |
| ai_jobs | city | Posting city; literal Remote deterministically identifies Remote target |
| ai_jobs | remote_type | Remote, Hybrid, Onsite; work arrangement target or known input for other tasks |
| ai_jobs | experience_level | Entry, Mid, Senior; seniority target or known input for salary/skills |
| ai_jobs | min_experience_years | Advertised minimum years; exact seniority proxy (0/2/5), excluded |
| ai_jobs | salary_min_usd | Advertised annual minimum in USD; target construction only |
| ai_jobs | salary_max_usd | Advertised annual maximum in USD; target construction only |
| ai_jobs | employment_type | Full-time only; constant and excluded |
| ai_jobs | posted_year | Year label, 2020–2026; temporal partition and numeric feature |
| ai_jobs | company_size | Small, Medium, Large |
| skills_demand | job_id | Claimed FK; mostly unmatched; no positional repair |
| skills_demand | skill | Required skill name; 11-label vocabulary |
| skills_demand | skill_category | Broad Programming/ML/Cloud grouping; excluded as target derivative |
| skills_demand | skill_level | Required Basic/Intermediate/Advanced proficiency; not applicant proficiency |
| country_ai_trends | country, year | Unique aggregate key; not joined to predictors |
| country_ai_trends | total_ai_jobs | Supplied aggregate count, inconsistent with source postings |
| country_ai_trends | avg_salary_usd | Supplied aggregate salary, separate from computed job means |
| country_ai_trends | remote_percentage | Supplied remote percent, 0–100 scale |
| country_ai_trends | top_skill | Aggregate top skill; not a per-job label |
| job_title_mapping | job_title | Unique title dimension key |
| job_title_mapping | standardized_title | Canonical title, identical to six source titles |
| job_title_mapping | role_category | Deterministic title category; not a valid ML target when title is given |
| data_dictionary | file, column, description | Publisher-provided documentation metadata |
| processed/jobs | salary_mid_usd | `(salary_min_usd + salary_max_usd) / 2`; salary regression target |
| processed/jobs | salary_width_usd | `salary_max_usd - salary_min_usd`; additional regression target |

Job-level midpoint is neither an observed accepted salary nor an individual employee's salary. Skills are posting demand, not skills possessed by applicants. Remote classification is a three-class task. The time range represents supplied year labels; verified collection dates are absent.
