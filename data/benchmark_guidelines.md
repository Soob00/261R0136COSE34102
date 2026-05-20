# Counterfactual Benchmark Guidelines

## Purpose

This document defines the common rules for constructing and curating Korean counterfactual evaluation sets for hate/offensive language detection in this project.

It is intended to be reused for:

- extending [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json)
- revising older sets such as [stereotype_flip_v2.json](/C:/nlp_project/data/stereotype_flip_v2.json)
- generating new candidate pairs with an LLM
- human validation and final benchmark release

## Current Datasets

- [kold_v1.json](/C:/nlp_project/data/kold_v1.json)
  - Main train/validation corpus.
- [hard_negative_testset.json](/C:/nlp_project/data/hard_negative_testset.json)
  - Larger candidate pool with multiple hard-case categories.
- [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json)
  - Curated benchmark used for the latest evaluation.

## What v3 Actually Is

`v3` was not directly generated from scratch by an LLM in this workspace.

The actual workflow used for `v3` was:

1. Start from the broader candidate pool in [hard_negative_testset.json](/C:/nlp_project/data/hard_negative_testset.json).
2. Manually select pairs that are more semantically stable and more plausibly label-preserving.
3. Reorganize them into three clearer categories:
   - `adversarial_clean`
   - `stereotype_flip_strict`
   - `subtle_stereotype`
4. Save the curated subset as [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json).

So the benchmark creation process is currently:

- candidate generation/pool construction
- curation
- category assignment
- validation

not pure end-to-end automatic generation.

## Core Design Principles

These principles apply to every future benchmark version.

1. Minimal edit
- Change as little as possible besides the identity term or target phrase.
- Avoid changing extra predicates, justifications, or consequences unless absolutely necessary for grammaticality.

2. Plausible label preservation
- After substitution, the gold label should still make sense.
- If the label changes after substitution, the pair is invalid as a counterfactual consistency example.

3. Semantic stability
- The pragmatic force of the sentence should remain comparable.
- Do not introduce new concrete risks, qualifications, or socially normal concerns that change the interpretation.

4. Category-aware design
- Do not mix all hard cases into one bucket.
- Separate at least:
  - non-offensive clean identity contexts
  - explicit stereotype/exclusion attacks
  - subtle stereotype or concern framing

5. Consistency is not enough
- A valid benchmark must support both:
  - consistency analysis
  - correctness/accuracy analysis
- A model can be consistently wrong.

6. Human review is required
- Final benchmark entries should be independently checked by at least 2 reviewers whenever possible.

## Inclusion Criteria

A pair can be included if all of the following hold.

1. The original sentence is fluent Korean.
2. The counterfactual sentence is fluent Korean.
3. The key framing remains comparable between original and counterfactual.
4. The expected label is defensible for both sentences.
5. The pair belongs clearly to one evaluation category.
6. The pair tests a meaningful failure mode rather than random lexical variation.

## Exclusion Criteria

Discard a pair if any of the following are true.

1. The substitution changes the social meaning too much.
2. The substitution introduces a new reasonable concern that weakens the original framing.
3. The original and counterfactual require different gold labels.
4. Too many words besides the target identity are changed.
5. The sentence becomes unnatural or obviously templated.
6. The pair duplicates an already covered phenomenon with no additional value.

## Category Definitions

### `adversarial_clean`

Expected label: `0`

Goal:
- Test whether the model produces false positives merely because an identity term appears.

Characteristics:
- neutral, supportive, descriptive, rights-based, or reporting context
- no attack on the target group

Good examples:
- support, rights, inclusion, anti-discrimination, neutral reporting

Bad examples:
- sentences containing negative insinuation, concern framing, or exclusion logic

### `stereotype_flip_strict`

Expected label: `1`

Goal:
- Test whether the model changes predictions when the target group is swapped in clearly offensive stereotype/exclusion frames.

Characteristics:
- explicit exclusion, broad derogatory generalization, competence denial, threat framing, group blame
- counterfactual should remain offensive after substitution

Good examples:
- "X should not be hired because ..."
- "X are inherently unfit / dangerous / inferior ..."

Bad examples:
- pairs where the substituted target introduces an ordinary, situational concern instead of group prejudice

### `subtle_stereotype`

Expected label: `1` in the current v3 design, but this category requires the most careful human review.

Goal:
- Test whether the model detects indirect, concern-based, or socially softened stereotype framing.

Characteristics:
- no explicit slur
- often framed as "concern", "adjustment", "social reaction", or "it may be difficult"
- may overlap with microaggression or polite exclusion

Warning:
- This category is the most fragile for label validity.
- Always review whether the pair is truly offensive under the KOLD task definition.

## JSON Schema

Each pair should follow this structure:

```json
{
  "id": 1,
  "category": "adversarial_clean",
  "original": "...",
  "counterfactual": "...",
  "expected_label": 0,
  "identity_orig": "조선족",
  "identity_cf": "한국인",
  "trap": "optional short note",
  "source_id": 1
}
```

Notes:

- `id`: benchmark-local running id
- `category`: evaluation category
- `expected_label`: shared label expected for both sentences
- `trap`: short explanation of the intended shortcut/failure mode
- `source_id`: origin id from a larger pool if applicable

## Human Validation Rubric

For each pair, reviewers should answer:

1. Fluency
- Are both sentences natural Korean?

2. Minimality
- Was the edit limited mostly to the target identity or target phrase?

3. Meaning preservation
- Does the substitution preserve the intended framing?

4. Label validity
- Is `expected_label` correct for both sentences?

5. Category fit
- Does the pair clearly belong to its assigned category?

Suggested decision labels:

- `accept`
- `revise`
- `reject`

## Recommended Workflow For Future Versions

1. Generate or collect a larger candidate pool.
2. Filter obvious failures automatically.
3. Manually review label preservation and category fit.
4. Keep source ids and edit history.
5. Evaluate baseline and comparison models.
6. Remove categories or pairs that produce misleading interpretations.

## Reporting Recommendations

When writing the paper, explicitly report:

- how candidate pairs were generated
- how many were rejected
- category counts
- whether labels were validated by humans
- whether consistency and accuracy were both measured

This is especially important because benchmark design strongly affects the conclusions.
