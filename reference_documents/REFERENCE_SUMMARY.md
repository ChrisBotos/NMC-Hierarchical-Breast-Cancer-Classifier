# Reference Documents Summary

Concise summaries of all reference documents for the CATS assignment. Claude must read this file for context on writing, grading, and submission standards.

---

## 1. CATS_instructions_2026-1.pdf

**The main assignment specification.**

- **Task:** Train a classifier to predict breast cancer subtypes (HER2+, HR+, Triple Negative) from aCGH copy-number data.
- **Data:** 100 labelled training samples (Train_call.txt + Train_clinical.txt), 57 unlabelled validation samples (Validation_call.txt). 244K-probe aCGH platform, preprocessed into segmented regions with CN calls (-1, 0, 1, 2). Coordinates are hg18/NCBI36.
- **Subtypes:** HER2+ (HER2 positive), HR+ (ER+ and/or PR+, HER2-), TN (ER-, PR-, HER2-).
- **Pipeline steps:** (1) Data purification/transformation, (2) Feature selection, (3) Choose classifiers, (4) Train and validate.
- **Language:** Python recommended (scikit-learn, pandas, numpy). Save model with joblib as .pkl.
- **Predictions format:** TSV, two columns `"Sample"` `"Subgroup"`, 58 lines (1 header + 57), .txt extension.
- **Validation command:** `python3 run_model.py -i unlabelled_samples.txt -m model.pkl -o output.txt`
- **Submission structure:**
  - `results/`: estimate.txt (number 0-57), prediction.txt
  - `model/`: model.pkl, run_model.py
  - `code/`: all scripts
- **Paper:** 5-6 pages A4, Bioinformatics journal style. Sections: Abstract (max 200 words), Introduction (research question, aCGH background, biomedical context, related work), Methods (classifier, CV scheme), Results (performance, single best biomarker, improvements after draft), Discussion + Conclusion (max half page), Tables & Figures (2-4 total), Author contributions.
- **Presentation:** Exactly 4 slides, max 8 minutes: (1) CV/testing scheme, (2) accuracy table, (3) methods used, (4) rationale for best method.
- **Grading:** Report 70%, Predictions+code 15% (format 5%, quality 5%, estimate 5%), Presentation 15%.

---

## 2. Research Project Grading Rubric MSc BSB.pdf

**The rubric used to grade the final paper.** Scale: fail / insufficient / sufficient / satisfactory / good / excellent. Dutch grades: 3.0 (fail) to 9.0 (excellent).

Graded items and what "excellent" looks like:

- **Abstract:** Basic description of research question, background, results, potential impact. Emphasizes key aspects.
- **Introduction:** Sketches context and methodology, reviews most important literature, states research question explicitly, provides novel connections between existing literature.
- **Methods:** Description of methodology with appropriate references. Work is reproducible. Provides insight into choices made. Complex methodology explained precisely, accurately, and insightfully. Helpful overview (e.g. workflow) available.
- **Results (main text):** Full results shown, clearly described. Student provides new insights. Complex results precisely and insightfully explained.
- **Figures and tables:** Appropriate amount (<10). Captions describe figures. Figures give full overview of results and provide insight in original or creative manner.
- **Discussion:** Insightful, shows opinion forming using current literature and results. Makes links beyond initial scope of project.
- **Writing style:** Scientific style with clear, precise, concise explanations. Clear structuring of introduction, methods, results, discussion. Excellent readability, ready for journal submission.

---

## 3. Scientific Writing Lecture (fob_lec9_2017_ScientificWriting.pdf)

**Key principles for the paper:**

### Writing fundamentals
- Write for a fellow MSc/BSc student, not your teachers.
- Five pillars: be **complete**, **precise**, **concise**, **clear**, and **interesting**.
- Complete: someone should be able to reproduce your experiment from the paper.
- Precise: give quantities and references; avoid vague language ("loads of" -> "630 million").
- Concise: cut every word that is not strictly necessary.
- Clear: highlight important parts, create a story line, one main point + ~5 supporting points.
- Interesting: write a story around your research question; use the abstract to set the story line.

### Style rules
- Mix active and passive voice (not exclusively active — avoid "we did this, we did that").
- Use present simple or past simple consistently. Avoid present continuous and past perfect.
- Tense consistency within sections.

### Paper structure
- **Abstract:** Motivation & research question, methods & results, conclusion & impact. Max 200 words. Aim at a wide audience.
- **Introduction:** Funnel-shaped — start general, zoom into specific research question. Include literature references. State research question explicitly.
- **Methods:** Allow reproducibility. Be precise with technical detail (parameters, equations). Include a flow chart.
- **Results:** Recipe: "In order to test X, we performed Y. Figure Z shows... This explains/is surprising/as expected."
- **Discussion:** Discuss issues affecting results, answer research question, compare to literature, explain impact on future research.
- **Figures/Tables:** Explain all axes/labels/lines in captions. Refer to each figure in main text and explain what can be seen.

### Common mistakes to avoid
- Plagiarism: read and reformulate, never copy. Use quotes + citations when needed. Always cite primary sources (not Wikipedia).
- Missing or poorly formulated research question.
- Conceptual misunderstanding of technical terms (use terms precisely).
- Sloppy wording.

### Peer review criteria
- Is the research question clear?
- Is the story line clear?
- Is it clear how the research was performed?
- Is it clear why this research question is interesting?

---

## 4. Authorship (PLOS ONE guidelines)

**How to write author contribution statements.** Use the CRediT Taxonomy roles:

| Role | Definition |
|------|-----------|
| Conceptualization | Ideas; formulation of research goals and aims |
| Data Curation | Annotate, scrub, maintain research data |
| Formal Analysis | Statistical, mathematical, computational analysis |
| Investigation | Performing experiments or data collection |
| Methodology | Development or design of methodology; creation of models |
| Project Administration | Management and coordination of research |
| Resources | Provision of study materials, computing resources, tools |
| Software | Programming, software development, testing |
| Supervision | Oversight and leadership |
| Validation | Verification of reproducibility of results |
| Visualization | Data presentation and visualization |
| Writing - Original Draft | Writing the initial draft |
| Writing - Review & Editing | Critical review, commentary, revision |

Each group member's contributions should be listed using these roles.

---

## 5. Background Reading (Canvas page)

**Recommended literature for the assignment:**

### CNV data preparation (context for understanding the preprocessed data)
- van de Wiel et al. (2011) — Preprocessing and downstream analysis of microarray DNA copy number profiles.
- van de Wiel et al. (2007) — CGHcall: calling aberrations for array CGH tumor profiles.
- van de Wiel & van Wieringen (2007) — CGHregions: dimension reduction for array CGH data.
- Venkatraman & Olshen (2007) — A faster circular binary segmentation algorithm for array CGH data.

### Machine learning (key references for methodology)
- **Wessels et al. (2005)** — A protocol for building and evaluating predictors of disease state based on microarray data. **Key concept: cross-validation.**
- "A guide to machine learning for biologists" — general ML reference.
- **Haury et al. (2011)** — The Influence of Feature Selection Methods on Accuracy, Stability and Interpretability of Molecular Signatures.
- **van 't Veer et al. (2002)** — Gene expression profiling predicts clinical outcome of breast cancer. *Directly relevant example of tumor classification.*

### General resources
- Wilson et al. (2017) — "Good enough practices in scientific computing" (PLOS Comp Bio). Project organization and manuscript writing.
- Manchester Phrase Bank — scientific writing resource.
- Hastie et al. — Elements of Statistical Learning (textbook for ML background).
