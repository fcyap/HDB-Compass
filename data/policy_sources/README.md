# HDB policy corpus: how it is built

The policy RAG answers from **official pages, kept word for word**. Each page becomes one document, so every
retrieved chunk cites the exact page it came from, the date it was captured and a hash of the saved copy.
Nothing is paraphrased or summarised by a person or a model.

| Step | Command / action | Output |
|---|---|---|
| 1. List the pages | edit `sources.yaml` | one entry per page: `id`, `url`, `publisher`, `policy_category`, `capture` |
| 2a. Fetch script-friendly pages (CPF) | `.venv/bin/python -m data.policy_sources.fetch_sources` | `raw/<id>.html`, `raw/manifest.json` |
| 2b. Save the other pages by hand (HDB, MAS, IRAS) | browser, see below | `raw/<id>.html` or `.pdf` |
| 2c. Record the hand-saved pages | `... fetch_sources --record-manual` | manifest entries (`method: manual`) |
| 3. Extract verbatim text | `.venv/bin/python -m data.policy_sources.extract_sources` | `extracted/<NN>_<id>.md` |
| 4. Check the facts | edit `evaluation/policy_fact_sheet.yaml`, run `pytest tests/test_policy_corpus.py` | verified facts with exact quotes |
| 5. Install as the corpus | `... extract_sources --install` | `data/policy_documents/` (old documents move to `data/policy_documents_legacy/`) |
| 6. Re-index | `.venv/bin/python data/ingest_hdb_policy_rag.py` | `data/policy_store.pkl`, `data/policy_chunks.json` |

Check progress at any time with `... fetch_sources --status`.

## Saving a page by hand (step 2b)
HDB's and MAS's websites refuse automated requests. That is their bot protection, and we respect it: a person opens
each page in a normal browser instead. For every `capture: manual` entry in `sources.yaml`:

1. Open the `url` in Chrome or Safari. If HDB has moved the page, find it from hdb.gov.sg and **correct the `url`
   in `sources.yaml`**. The listed HDB URLs are best guesses that could not be checked by script.
2. Check that the page is the one the `notes` describe, and that it has fully loaded (expand any collapsed
   "accordion" sections, since collapsed text may not be saved).
3. Save it into `data/policy_sources/raw/`, either as **`<id>.html`** or under the browser's own file name (the
   page title) with that name in the entry's `file:` field:
   - Chrome: File → Save Page As → Format **"Webpage, HTML Only"**;
   - Safari: File → Save As → Format **"Page Source"**.
   
   If the saved HTML extracts badly (step 3 reports fewer than 80 words), save it instead as **`<id>.pdf`**
   (File → Print → Save as PDF).
4. Set `verified: true` on that entry in `sources.yaml`.

Then run step 2c. A saved file that is really an error or bot-check page is rejected with a message.

### What the extractor does with HDB pages
HDB's site (Next.js on Sitecore) renders its text with JavaScript, so an "HTML Only" save holds the article in a
`__NEXT_DATA__` JSON block rather than in the page body. The extractor reads that block:
- the main rich-text fields, then each accordion section as a `### title` section;
- the page's own address from its route, which is why the `url` of every HDB entry comes from the saved file itself.

It also:
- strips bold markup from headings;
- drops footnote markers (`<sup>[1]</sup>`);
- merges HDB's two-row table headers ("Classification of flats" above "Standard | Plus | Prime");
- moves footnote rows out of tables as a `Notes:` paragraph;
- drops site banners that repeat on 3 or more pages of one publisher.

Wording is never changed. Pages that only link to their sub-pages are listed with `exclude: true`.

Only a date the page itself states is kept (`Page Date`). Most HDB pages state none, and a date found elsewhere in
the text is usually a policy's history ("from 3 March 2006"), so it is not used. Freshness is tracked by `Retrieved`.

## Checking facts (step 4)
`evaluation/policy_fact_sheet.yaml` is the ground truth for policy answers. Facts marked `pending_capture` hold a
`claim_to_check`: what the old corpus or a team note says, which is **not** truth yet. Once the page is saved:

- copy the exact sentence from the page into `quote`;
- write the `fact` as the page states it (correct it if it differs from the claim);
- set the `key_clause` words an answer must contain, and set `status: verified`.

The test then confirms the quote really is on that page. Settle the two known-wrong figures first: Plus and Prime
subsidy recovery.

## Keeping it current
- `fetch_sources --refresh` re-fetches the CPF pages and prints **CHANGED** for any page whose content hash differs.
- Re-save the HDB pages by hand at the same time (every few months, or after a Budget / National Day Rally housing
  announcement).
- Re-run steps 3–6 afterwards. Every chunk carries its page's `Retrieved` date, and answers show it.
