DECISIONS LOG
KV-cache eviction and planted customer complaints

How to read this log
Each entry says what was decided, who decided it, and why.

ABOUT PRE-REGISTRATION

No pre-registration was written before the runs. I worked through the task as given (starting with the four tests) and did not write one.

These rules came from the task itself and were fixed before any results existed:
  - Retention cut-off: tau = 0.2. Below it, the planted review counts as "thrown away".
  - Calibration rule: change N_RECOMMENDATIONS once (5 to 8, or 5 to 3) only if Full mentions the complaint in under 30% or over 95% of calibration runs.
  - Verdict rule: at least 2 of the 3 eviction methods drop by 10 points or more against Full, while their share of well-formed answers stays within 5 points of Full's.

This rule was decided after labelling:
  - Detector choice: the detector with the highest F1 against the labels. The keyword detector would have won under precision, recall or F1.


STAGE 1. SET-UP

1. Hardware: a Colab T4 GPU.
   Effect: the model is Qwen2.5-1.5B-Instruct in float16, because a T4 has no bfloat16.

2. The data was built on my laptop.
   Why: uploading a few MB of prepared data instead of a 4 GB Yelp zip.
   Claude fixed a Windows text-encoding problem so this would run.


STAGE 2. BUILDING THE DATA

3. Which restaurants qualify.
   Who: Claude.
   Rule: scan 80 random open restaurants with a seed, and keep the first 12 that have at least 40 English reviews of 120 to 260 tokens, and where none of the 30 sampled reviews already mentions any of the three complaints.
   "English" means at least 97% ASCII characters and at least 3 very common English words.
   Rejected restaurants are listed in logs/data_build.log.

4. Balancing.
   Who: Claude, following the task.
   Restaurant i gets complaint type (i + position) mod 3 and template i mod 2, so every complaint type appears at every position equally often. Plus one control per restaurant: 48 sets in total.

5. Length matching.
   Who: Claude.
   Each planted complaint is padded with neutral sentences to within 15% of that restaurant's median review length, so it doesn't stand out.

6. Word lists and the six fake complaints (lexicon.py, templates.py).
   Who: Me, with chatgpt to generate rubbish text.
   Both were frozen before running.

7. Licence.
   Who: Claude.
   Raw review text stays in the git-ignored data/ folder. redact.py replaces any stretch of 60 or more characters copied from a real review with "[redacted review text]", and checks that no detector result changes.

8. How retention is averaged.
   Who: Claude.
   For each token, the share of (layer, KV head) pairs that kept it, then the mean over the review's tokens.

9. The yes/no question on controls.
   Who: Claude.
   A control has no planted complaint, so it is asked about the complaint its restaurant gets at review 15. A "yes" there counts as a false positive.


STAGE 3. TESTS AND TIDYING

10. The four required tests.
    Who: Me, out of the four, three were written by me and reviewed by Gemini, while test_license.py was edited a lot by claude- as my code was buggy and I had no direction as to how to proceed. I don't know how json works so parsing it wasn't something I had done before.

11. Code trimmed.
    Who: Claude, at my request.
    Why: the code was too long to explain line by line. Extras the task did not ask for were removed (about 740 to about 610 lines of code). Claude checked that the trimmed code builds a byte-for-byte identical manifest and gives identical tables.


STAGE 4. FIRST RUN ON COLAB

12. Double chat template fixed before any result was saved.
    Who: Claude (the bug was in Claude's code).
    What: render() wrapped the reviews in the chat template, and the kvpress pipeline wrapped them again. A safety check in run.py compares our tokens with kvpress's own preprocessing; it stopped the first calibration run before anything was written. After the fix, all 48 contexts match kvpress token for token.


STAGE 5. CALIBRATION

13. Answer limit raised from 256 to 512 tokens.
    Who: Me.
    Why: at 256, every calibration answer was cut off, and many were stuck listing "Review 13, Review 14...", which made the citation detector look better than it was. No real runs existed yet.

14. Calibration rule applied at 512 tokens.
    Who: Me.
    Result: Full mentioned the complaint in 4 of 12 calibration runs (33%), inside the 30% to 95% band, so N_RECOMMENDATIONS stayed at 5.


STAGE 6. THE REAL RUNS

15. 240 runs: 48 sets x 5 methods.

16. Not raising the answer limit to 1024.
    Who: Me, after finding that 25 of 48 Full answers hit 512 tokens.
    Why: raising it would mean redoing all 240 runs after seeing results, and the loops would just get longer.


STAGE 7. LABELLING AND DETECTOR CHOICE

17. The labelling sheet.
    Who: Claude designed it (labels.py).
    36 answers: 6 planted answers per method plus 6 controls, shuffled, with the method and condition hidden. Every row has all three complaint columns, so the same rule applies to every row.

18. The labels themselves.
    Who: Me. All 36 rows were labelled by me in labels/to_label.csv (made by labels.py), saved as labels/labels.csv.
    Rule: 1 if the advice states the specific problem, in any words, or recommends fixing it. 0 if it only gives a review number, names a related but different issue, or is too vague to tell.
    8 broken outputs (empty, repeated characters, or long lists of review numbers) were noted and labelled 0.
    Borderline rows were noted: L14 and L32 (parking, no towing stated) and L15 (disabled access through parking).
    Final totals: access 4 rows, towed 1 row, double charge 0 rows.

19. L14 (towed): set to 1 at first, then corrected to 0.
    Who: Me.
    What happened: I first set L14 to 1 against my own judgement. Before finalising the labels, I ran the analysis with L14 at both 1 and 0 and saw the detector scores change noticeably. I set it to 0 because that is what my written rule and my judgement say; a 1 would have been false data.
    Deviation: running the analysis before this label was final breaks strict blind labelling, so I report it here.

20. Second reader.
    Who: Claude, with my supervisor's approval; I changed nothing because of it.
    Claude read all 36 rows and, with L14 at 0, agreed on every cell.

21. Detector choice: keyword.
    Who: Me, from the labels, by highest F1.
    Result on the 30 planted rows: keyword found 5, with 0 false alarms and 0 misses. The citation detector raised 5 false alarms in 8 flags, fooled by answers that list review numbers.

Only 5 of the 30 labelled planted answers mentioned the complaint, so the detector comparison rests on few cases. Across all 180 planted runs, the keyword detector found a mention in 29.
