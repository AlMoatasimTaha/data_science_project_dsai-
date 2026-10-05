# Target discovery and leakage decisions

| Question | Decision | Available prediction-time inputs | Main metrics |
|---|---|---|---|
| What salary midpoint does a posting advertise? | Train | Title, organization type/size, industry, country, year, known experience and work arrangement | MAE USD, RMSE USD, R², interval coverage |
| Can seniority be inferred before years and salary are known? | Train, possible no-signal result | Base posting fields and work arrangement | Macro F1, balanced accuracy, accuracy, Brier |
| Can work arrangement be inferred without the location shortcut? | Train, possible no-signal result | Base posting fields and known experience | Macro F1, balanced accuracy, confusion matrix |
| Which skills are demanded by a posting? | Restricted exploratory experiment | Exact-key matched jobs and independent posting attributes | Micro/macro F1, Hamming loss, subset accuracy, average precision |
| Is advertised salary range width predictable? | Additional valid task | Same input availability as salary midpoint; no endpoints | MAE USD, RMSE, R² |
| Can years identify experience? | Reject as tautology | Years map exactly to labels | Not a useful predictive problem |
| Can city identify Remote? | Reject shortcut | Remote city token reproduces target | Not valid before work arrangement is known |
| Can standardized title or role category be predicted from title? | Use dictionary mapping | Exact deterministic title mapping | No ML needed |
| Can employment type be predicted? | Reject | Single observed label | No learnable target |
| Can national AI employment be forecast? | Reject for reliable ML | Seven annual rows per country; unreconciled counts | Descriptive inspection only |
| Can missing skills be inferred as empty skill sets? | Reject | Missing relationships are unknown, not negatives | Would fabricate labels |
| Can descriptions support embeddings/NLP? | Reject | No description column exists | Compare title TF-IDF only |

The five trained tasks are feasible formulations, not promises of learnable signal. Results that fail baseline gates remain explicitly experimental in the application and API. Salary prediction assumes the posting's seniority is already known. Chaining an uncertain seniority prediction into salary is not evaluated and is not advertised.

The schema's skill vocabulary is fixed before splitting; prevalence, decision thresholds, cardinality baseline and preprocessing are fit on training rows only. Per-skill decisions use 0.5 for the learned models; the prior baseline predicts its train-derived top-k labels. No holdout threshold optimization occurs.
