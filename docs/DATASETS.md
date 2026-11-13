# Datasets

Last reviewed: 2026-11-13. Rule: results are always reported with the dataset name, its real/synthetic
status, and its license. Nothing here has been downloaded yet.

## Decisions (Phase 2.1)

| Task | Dataset | Real or synthetic | License | Status |
|---|---|---|---|---|
| Intent classification (primary) | [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | **Synthetic / hybrid** (card: "hybrid synthetic dataset") | `cdla-sharing-1.0` | Selected |
| Intent classification (secondary benchmark) | [PolyAI Banking77](https://huggingface.co/datasets/PolyAI/banking77) | Collection method **not stated** on the card | `cc-by-4.0` | Selected (optional) |
| Sentiment analysis | [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (subsample) | Real reviews; labels are **weak** (from star ratings) | No formal license; maintainers state research use | Selected |
| Ticket priority, routing, resolution time | none yet | - | - | **Deferred** |

## Details

### Bitext (intent)
- ~26.9k rows; columns `flags`, `instruction`, `category`, `intent`, `response`; 27 intents in 11 categories (per viewer header).
- Includes e-commerce intents we need (e.g. `cancel_order`, `track_refund`, `check_refund_policy`, `complaint`).
- Because it is synthetic, metrics measure how well a model learns this dataset's phrasing, **not** real-world customer performance. Always label results "synthetic data".
- CDLA-Sharing-1.0 has share-alike terms for shared data. I have not read the license text; read it before redistributing the data or derived datasets. Trained models and code are a separate legal question - not assessed here.

### Banking77 (intent, optional)
- 13,083 queries, 77 intents; 10,003 train / 3,080 test. Banking domain, so it is a robustness check, not our target domain.
- Cite: Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders" (https://arxiv.org/abs/2003.04807).

### Amazon Reviews 2023 (sentiment)
- 571M reviews; we will use a small subsample (category and size decided in Phase 2.2).
- Labels: derived from star ratings (proposed: 1-2 negative, 3 neutral, 4-5 positive). This is weak supervision, and product reviews differ from support messages. Report this limitation with any score.
- No formal license is attached; the maintainers say it is for research use. Do not redistribute the data.

### Deferred: priority, routing, resolution time
No candidate has a verified license or confirmed real labels. Candidates to inspect later: Kaggle support-ticket datasets (suraj520, suvroo) and Mendeley "Help Desk Tickets". If a chosen one turns out to be synthetic, models trained on it are demo-only.

## First look at the downloaded data (2026-11-13)

Downloaded to `data/raw/` (git-ignored); sizes match the Hugging Face file listing and hashes are in each `MANIFEST.json`.

### Bitext (26,872 rows; columns `flags`, `instruction`, `category`, `intent`, `response`)
- No missing values. 27 intents in 11 categories; each intent belongs to exactly one category.
- **Almost perfectly balanced**: 950-1000 rows per intent. This looks curated/synthetic, so accuracy on it will be high and says little about real traffic.
- Very short messages: median 9 tokens, p99 14, max 17.
- **Duplicates**: 2,237 exact duplicate `instruction` values (2,598 after cleaning). There are **no** label conflicts. => Deduplicate on the cleaned text **before** splitting, or the same sentence lands in both train and test and inflates scores.
- **Placeholders**: 6,670 rows (about 25%) contain `{{...}}` tokens; 9 distinct (e.g. `{{Order Number}}` 2,907 rows, `{{Account Type}}`, `{{Person Name}}`, `{{Refund Amount}}`). Currently `tokenize` would split them into plain words, which loses that they are placeholders.
- Noise: typos (e.g. "newslettert") and inline ids like `#00108`. No URLs or emails.
- `flags` has 394 distinct codes (the card's meaning of the letters is not verified here). Not used as a feature.
- No sentiment or priority labels in this dataset.

### Banking77 (train 10,003 / test 3,080; columns `text`, `label` as integer id)
- No missing values. 77 classes in both splits; **imbalanced**: 35-187 per class (median 127).
- Messages: median 10 tokens, p99 43, max 79.
- 4 duplicates after cleaning in train; **7 texts appear in both train and test** (about 0.2% of test) - remove from train or report it.
- Label names are not in the parquet files (only ids); the id-to-name list must be taken from the dataset card.

### Decisions for the next step
1. Map `{{Placeholder Name}}` to a single token such as `<order_number>` in `clean_text` (keeps their meaning).
2. Deduplicate on cleaned text before any train/validation/test split; stratify by intent.
3. Use Banking77's official train/test split, drop the 7 overlapping texts from train, and get the label names from the card.
4. Keep reporting Bitext results as "synthetic data" and Banking77 as the less-controlled check.

## Open verification items
- [x] Licenses read by the user before downloading (CDLA-Sharing-1.0 and CC BY 4.0); the user's notes on implications are not recorded here.
- [ ] Confirm from Bitext's full card that the full intent list matches (only part of the page was read verbatim).
- [ ] Banking77 data provenance (not stated on the card).
- [ ] Amazon Reviews 2023: read the terms on the official site before downloading.

## How to download
```powershell
$env:PYTHONPATH = "src"
python -m supportx.data.download bitext --accept-license      # -> data/raw/bitext/
python -m supportx.data.download banking77 --accept-license   # -> data/raw/banking77/
```
- Without `--accept-license` the tool refuses and prints the license; read the license first.
- Files come from a pinned Hugging Face commit; `MANIFEST.json` records revision, license, the synthetic flag and SHA-256 hashes.
- Amazon Reviews 2023 is not automated: it is huge and its terms are still unread. A subsample step comes later (Phase 2.3).
- Synthetic demo data for tests: `python -m supportx.data.sample` (demo only; labels are rule-generated).

## Storage rules
- Real/downloaded data: `data/raw/` (git-ignored, never committed).
- Synthetic demo data for tests only: `data/sample/` (small, clearly labeled, may be committed).
