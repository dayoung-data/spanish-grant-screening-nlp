# Automated Screening Support for SME Grant Applications (Spanish NLP)

Transformer-based scoring pipeline that helps reviewers assess free-text grant applications on two criteria: **originality** (how differentiated an answer is from peers) and **concreteness** (how well claims are backed by evidence). Built for Sercotec, a Chilean government SME-support agency, during an IDB consultancy (Jun 2024 – May 2025).

> **Confidentiality.** The application texts, annotations and trained models belong to the client and are not in this repository. Column and file names have been renamed, notebook outputs that contained applicant data were removed, and `data/sample/` holds a small synthetic example of each data format.

## Why

Hundreds of applicants answer open-ended business-model questions, and several reviewers score them by hand. Reviewers read the same text differently, so scores drift. The system gives reviewers a shared, reproducible reference signal for two things they were already looking for. It does not replace their judgement.

## Architecture

```
                       applicant free-text answers
                                   |
              +--------------------+---------------------+
              |                                          |
      ORIGINALITY (unsupervised)               CONCRETENESS (supervised, 2 stages)
              |                                          |
  group answers by applicant x criterion      Stage 1  sentence classifier
  embed with gte-multilingual-base            RoBERTa-large-BNE (Spanish)
  mean vector per group                       CLAIM / EVIDENCE / OTHER
              |                               +/-4 sentence context window
  compare with peers in the same                         |
  economic activity + same criterion          Stage 2  claim <-> evidence cross-encoder
  originality = 1 - mean cosine sim           XLM-RoBERTa-large, "is this claim supported?"
              |                                          |
              |                               per answer: for every claim count supporting
              |                               evidence (p > 0.5), log1p, average over claims
              |                               = LogWeightedMatchRate
              |                                          |
              +------------- min-max normalise, average per applicant -----------+
                                   |
                         0-1 scores shown to reviewers
```

## Methods

### 1. Originality (`notebooks/originality/`)
- **Prep.** Questions are mapped to three criteria (product/service originality, customer-targeting differentiation, operating-model originality); answers are concatenated per applicant and criterion.
- **Embedding.** `Alibaba-NLP/gte-multilingual-base`, one vector per text, averaged into a representative vector per applicant x criterion.
- **Score.** `originality = 1 - mean cosine similarity` to all *peers* (same economic activity, same criterion). Texts under 150 tokens are not scored. Scores are averaged per applicant and min-max normalised.

### 2. Concreteness (`notebooks/concreteness/`)
**Stage 1: claim / evidence / other.** `PlanTL-GOB-ES/roberta-large-bne`, 3 classes. Each sentence is classified together with its +/-4 neighbouring sentences so the model sees local context. At inference, explicit `Title: content` blocks are extracted by regex and take priority. For the remaining sentences, a keyword and probability-threshold fallback relabels borderline "other" sentences.

**Stage 2: does evidence support the claim?** `xlm-roberta-large` cross-encoder on (claim, candidate support) pairs, trained as binary related / not related. Labels come from annotators' "fully supported / partially supported" judgements. Negatives are other supports taken from the *same answer*, which makes them harder than random pairs.

**Scoring.** For each answer, every predicted claim is paired with every predicted evidence sentence. Per claim, `log1p(#supports with p > 0.5)`; the mean over claims is the **LogWeightedMatchRate**, then min-max normalised and averaged per applicant.

#### Worked example

The concreteness score asks one question: *is this plan just words, or does each claim have evidence behind it?* Take this (invented) answer:

> We will buy an electric oven to cut energy costs. In a pilot, our gas bill dropped 18%. We want to be the neighbourhood's reference bakery. Since 2022 we have supplied 120 loaves a day to three cafes. The shop is called "Pan de Barrio".

**Stage 1** labels every sentence:

| Sentence | Label |
|---|---|
| We will buy an electric oven to cut energy costs. | Claim |
| In a pilot, our gas bill dropped 18%. | Evidence |
| We want to be the neighbourhood's reference bakery. | Claim |
| Since 2022 we have supplied 120 loaves a day to three cafes. | Evidence |
| The shop is called "Pan de Barrio". | Other |

**Stage 2** pairs every claim with every evidence sentence and asks whether the evidence supports the claim (a cross-encoder reads both sentences together):

| Claim | Evidence | Supported? |
|---|---|---|
| electric oven | gas bill dropped 18% | yes |
| electric oven | supplies three cafes | no |
| reference bakery | supplies three cafes | yes |
| reference bakery | gas bill dropped 18% | no |

**Score.** Each claim has one supporting sentence, so each contributes `log1p(1) = 0.69` and the answer scores **0.69**. An answer with the same two claims and no evidence scores **0**. A claim with six supporting sentences contributes `log1p(6) = 1.95`, which is higher but not six times higher, so repeating the same point does not inflate the score.

## Results

| Model | Data (this repo's notebooks) | Validation | Best result |
|---|---|---|---|
| Stage 1: claim/evidence/other | 494 context windows (444 train / 50 val) | random 10% split | macro-F1 **0.73**, accuracy 0.74 (epoch 6) |
| Stage 2: claim-support | 6,094 pairs, balanced to 5,300 (4,770 train / 530 val) | random 10% split | F1 **0.83**, accuracy 0.82 (epoch 5) |

Training curves are kept in the training notebooks (`02_` and `05_`).

## Design decisions

- **Peer-relative originality.** Similarity is only computed against applicants in the same economic activity, so a sector's shared vocabulary does not make a whole sector look unoriginal.
- **`log1p` on support counts.** One claim backed by six sentences is not six times better than one backed by a single strong sentence. The log keeps long, repetitive answers from dominating.
- **Context windows for Stage 1.** Whether a sentence is a claim or evidence often depends on its neighbours, so the classifier gets +/-4 sentences.
- **Rules plus model.** Applicants who already write in a `Title: content` structure are labelled deterministically. The model handles free prose.
- **Hard negatives.** Negative pairs come from the same answer, so the cross-encoder has to separate related from unrelated content within one applicant's own text.
- **Scores are relative.** Min-max normalisation makes scores comparable within one batch of applications, not across batches.

## Limitations

- Validation sets are small (50 and 530 examples), and the splits are random at the example level. Windows or pairs from the same applicant can land in both train and validation, so the reported numbers are probably optimistic. An applicant-level split is the first thing I would change.
- Human-agreement analysis (model scores vs. reviewer scores) is not included here.
- Keyword lists and thresholds in Stage 1 inference were tuned by hand on a small sample of applicants.
- The system supports reviewers; it was not designed to make funding decisions.

## Repository layout

```
notebooks/
  originality/
    01_data_preprocessing.ipynb          # question -> criterion mapping, grouping
    02_average_embedding_originality.ipynb
  concreteness/
    01_claim_evidence_dataset.ipynb      # annotated spans -> context-window dataset
    02_claim_evidence_training.ipynb     # Stage 1 training
    03_claim_evidence_inference.ipynb    # Stage 1 inference + rule-based correction
    04_support_pairs_dataset.ipynb       # annotated claim-support -> pair dataset
    05_support_matching_training.ipynb   # Stage 2 training
    06_support_matching_inference.ipynb  # LogWeightedMatchRate per answer / applicant
data/sample/                             # synthetic examples of each data format
requirements.txt
```

Notebooks were developed on Google Colab (GPU). Run them in the order of their numeric prefixes. Paths such as `data/...` and `models/...` are placeholders to point at your own data. Comments are mostly in Spanish, since that is the language of the data and the client.

## My role

Project lead and sole developer: planning, data design, modelling, deployment and evaluation. I coordinated a team of 25 annotators who labelled the training data. I also ran workshops for Sercotec staff on working with AI-assisted tools. The main outcome for the client was a shared reference that let multiple reviewers assess applications against more consistent criteria.
