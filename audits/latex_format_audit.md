# LaTeX / format audit: paper_submission_revision (2026-10-03)

- Official sources: raw.githubusercontent.com/JmlrOrg/tmlr-style-file/main/{tmlr.sty, tmlr.bst, fancyhdr.sty, math_commands.tex, main.tex}, at repo HEAD 7bf90ef (2023-06-30).
- The repo contains no `tmlr.tex`. The official example file is `main.tex`.
- Local copies were not modified. The scratch rebuild was done with tectonic in `/tmp/claude-2011/tmlr/build`. It gives the same 34 pages and the same heading pages as the committed `main.pdf` (pdfTeX 1.40.29). The committed `main.pdf` and `main_tmlr_submission_revision.pdf` are byte-identical.

## 1. Style-file diffs (`diff official local`)

| File | Result | sha256 (first 16 hex chars) |
|---|---|---|
| tmlr.sty | IDENTICAL | 816214ff5919aa45 |
| tmlr.bst | IDENTICAL | 306fd454cf40771b |
| fancyhdr.sty | IDENTICAL | 3d2922548e0e5f1a |
| math_commands.tex | IDENTICAL (and not `\input` by local main.tex, which is allowed) | 90473c4d0542070d |

## 2. Geometry, verified from the official `tmlr.sty` and measured in the PDF

| Parameter | Official tmlr.sty | Measured (main.pdf) | OK |
|---|---|---|---|
| Paper | `\paperwidth 8.5in`, `\paperheight 11in` (US Letter) | 612 x 792 pt | yes |
| Text width | `\textwidth 6.5 true in` (468 pt), `\oddsidemargin 0in` | text x-range 72.0–540.2 pt (468 pt) | yes |
| Text height | `\textheight 9.0 true in` (648 pt), `\topmargin -0.625in`, `\headsep` +0.25in | body y-range 84.0–732.0 pt (648 pt); header at 27.6 pt; folio at 752 pt | yes |
| Body font | article 10pt, T1 and lmodern | dominant span size 10.0 pt on every page | yes |
| Header | review mode: `Under review as submission to TMLR` | present on every page | yes |
| Title block | review mode: "Anonymous authors / Paper under double-blind review" | rendered as such | yes |
| Overfull boxes | — | 0 in `main.log` | yes |

## 3. main.tex compared with the official example

| ITEM | CURRENT | EXPECTED (official) | Status / FIX |
|---|---|---|---|
| Document class | `\documentclass[10pt]{article}` | same | OK |
| tmlr option | `\usepackage{tmlr}` (no option, i.e. review/anonymous) | `\usepackage{tmlr}` for submission; `[accepted]` for camera-ready; `[preprint]` for arXiv | OK for submission |
| `\author` | `\name Anonymous \email anonymous@example.com \\ \addr Anonymous Institution` | Real `\name`/`\email`/`\addr` blocks separated by `\AND`. The sty hides them in review mode ("Authors ... should be hidden as long as the tmlr package is used without the [accepted] or [preprint] options") | Acceptable: nothing leaks. FIX at camera-ready: fill in real authors |
| `\month` / `\year` | `\def\month{08}`, `\def\year{2026}` | `MM` / `YYYY`, used only in accepted mode | Harmless. FIX at camera-ready: set to the acceptance month |
| `\openreview` | `\url{https://openreview.net/forum?id=XXXX}` | same placeholder | OK until camera-ready |
| Abstract | `abstract` environment, one paragraph (sections/00_abstract.tex) | "limited to one paragraph" | OK |
| Citations | natbib loaded by tmlr.sty (authoryear, round); only `\citep`/`\citet`, no bare `\cite` | same | OK |
| Bibliography | `\bibliographystyle{tmlr}` then `\bibliography{references}` | `\bibliography{main}` then `\bibliographystyle{tmlr}` (order irrelevant) | OK |
| Broader Impact | `\subsubsection*{Broader Impact Statement}` before references | same position and heading | OK. Optional, since the work carries no significant risk of harm |
| Reproducibility Statement | `\subsubsection*{Reproducibility Statement}` before references | not in template; not forbidden | OK, but it counts toward main-body length (see Section 6) |
| Author Contributions / Acknowledgments | absent | template has them as optional `\subsubsection*` | OK for anonymous review; add at camera-ready |
| **LLM disclosure footnote** | **none** (the manuscript contains no `\footnote` at all) | FAQ: "we also require that authors mention explicitly in their submission that they have used this tool, as a footnote on the first page." | **FIX if any LLM was used** in writing, code or analysis: add a first-page footnote, e.g. in the abstract or the first paragraph of the Introduction, stating the tool and how it was used. Footnotes inside `\title`/`\author` are dropped in review mode, so do not put it there |
| Appendix | `\clearpage\appendix` after bibliography; sections A–I | `\appendix` after bibliography | OK (`\clearpage` is cosmetic) |
| Extra packages | amsmath/amssymb/amsfonts/amsthm, graphicx, booktabs, multirow, hyperref, url | hyperref, url (others not forbidden) | OK. None of them changes the layout |
| geometry / fonts | no `geometry`, no font packages, no `\linespread`, no `\baselinestretch`, no `\setlength`/`\addtolength` of text dimensions, no `\enlargethispage`, no negative `\vspace`/`\vskip` | unmodified layout | OK |

## 4. Formatting hacks (grep of main.tex, sections/*.tex, tables/*.tex)

| ITEM | CURRENT | EXPECTED (official) | FIX |
|---|---|---|---|
| Negative `\vspace`/`\vskip` | none (only `\vspace{1em}` at tables/grid.tex:14) | — | none |
| `\setlength` / `\textheight` / `\textwidth` / `\baselineskip` / `\linespread` / `\enlargethispage` / `geometry` | none | — | none |
| `\small` | only inside `table` floats: sections/04_measurement.tex:53,116,165,221; 05_training.tex:38; 07_pitfalls.tex:74; 10_appendix.tex:15,51,348; 12_app_measurement.tex:70,132,217,272,302,365. None in body text | 10pt body | none needed. Table text at 9pt is standard practice |
| `\resizebox` scaled **up**: Table 3 `tab:precision` (sections/04_measurement.tex:174, p.9) | effective table font **12.5 pt**, larger than the 10pt body | not prohibited; no official rule | Cosmetic, inconsistent typography. Drop `\resizebox` and keep `\small`, or use `\resizebox{\linewidth}{!}` only when the table is wider than `\linewidth` (e.g. `adjustbox` with `max width=\linewidth`) |
| `\resizebox` scaled up: Table 2 `tab:autotune` (04_measurement.tex:125, p.8) | effective 10.4 pt | — | same as above (optional) |
| `\resizebox` scaled down: Table 8 `tab:backward` (12_app_measurement.tex:281, p.22) | effective 8.1 pt | no minimum stated | Readable. No action required |
| `\resizebox` scaled down: Table 9 `tab:trainstep` (12_app_measurement.tex:307, p.23) | effective 8.5 pt | — | Readable. No action required |
| `\resizebox`: Table 10 `tab:prescore` (12_app_measurement.tex:374, p.24) | effective 10.1 pt | — | none |
| Figure text | smallest glyphs 5.4 pt (Fig. 2 `fig:ksweep` tick exponents, p.8) and 5.9 pt (Fig. 1 `fig:taxonomy`, p.4); most labels 6.8–8.8 pt | no rule | Optional: raise the matplotlib tick and exponent font size to at least 7 pt |
| Unused file | figures/fig4_roofline.pdf is not included anywhere | — | Optional: drop it from the upload |

No item in this section alters the style's formatting, font or layout in the sense of the SUBM rule. Nothing here is a desk-reject risk.

## 5. Anonymity

| Check | Result |
|---|---|
| Author block in PDF | "Anonymous authors / Paper under double-blind review" (style-enforced) |
| PDF Info dict, main.pdf | Author, Title, Subject and Keywords all empty; Creator "LaTeX with hyperref"; Producer pdfTeX-1.40.29 |
| PDF objects and streams, main.pdf (decompressed scan for mhamdan, /home, Hamdan, SPARKNET, TopK) | No hits. `/PTEX.FileName` entries are relative (`./figures/fig1_taxonomy.pdf`, ...). The only "tarmac" hits are `cite.das2019tarmac` link anchors |
| Figure PDFs (fig1–fig5) | Matplotlib 3.10.9 metadata only; Author empty; no XMP; no path or username strings |
| Extracted text, main.pdf | No author names, institutions, emails or personal URLs. All URLs are bibliography URLs (arXiv, ACL, PMLR, NeurIPS, OpenReview) plus the `\openreview` placeholder (not rendered in review mode) |
| Self-citation | `hamdan2026marliotp` (Hamdan, Ghaleb, Bali, Cheriet; PRAI 2026) exists in references.bib:545 but is **not cited** and is absent from main.bbl. OK |
| Acknowledgments / funding | none |
| Code release statement | "provided as anonymised supplementary material and will be released publicly with the camera-ready version". OK. The supplementary ZIP must also be anonymized (AUTH); not audited here |
| **Resubmission cue** (low severity) | sections/10_appendix.tex:202 "the dense baseline **in this revision**". Script names `run_revision_campaign.sh`, `make_revision_tables.py`, `make_revision_figures.py` and `check_revision_numbers.py` appear at 10_appendix.tex:362,371,372,374. "an earlier version of this probe" at 10_appendix.tex:239 and "an earlier campaign" are also present. These do not identify the authors but tell reviewers this paper is a revision of a prior submission. FAQ: only the AE sees the previous-submission URL. FIX: replace "in this revision" with "in this paper"; optionally rename the scripts in the text and the supplement |
| Hardware detail | "one NVIDIA RTX PRO 6000 Blackwell ... of the two on a host with two Intel Xeon 6507P" and "shared with other tenants". Not identifying. OK |

## 6. Page counts (built PDF; headings located by PyMuPDF text extraction)

| Quantity | Value |
|---|---|
| Total pages | **34** |
| Sections 1–6 | Intro p.1, Related Work p.2, Cost Model p.3, Experiments p.6, Discussion and Limitations p.11, Conclusion p.12 |
| Broader Impact and Reproducibility statements | end of p.12, continuing onto p.13 |
| **References begin** | **p.13** (heading at y = 123 pt; body area starts at 84 pt) |
| References span | pp.13–15 |
| **Appendix A begins** | **p.16** ("A Proofs and assumptions") |
| Appendices | A p.16, B p.18, C p.25, D p.26, E p.28, F p.29, G p.31, H p.32, I p.33 (ends p.34) |
| **Main body (before References)** | **12 full pages + 2 lines on p.13** (≈ 12.03 pages). The 2 lines are the tail of the Reproducibility Statement: "in the text and fails on any disagreement. ... listed in Appendix I." |

**Implication:** the review-time rule is "up to 12 pages (of main content, before references)". This manuscript exceeds it by two lines, which puts it in the over-12-page track (4-week reviews). This is not a rejection criterion, since there is no hard limit.

FIX: cut at least 3 lines anywhere in pp.1–12 so the References heading starts at the top of p.13. Do not use layout hacks; trim text. For example, shorten the Reproducibility Statement, which is mostly restated in Appendix I.

## 7. Summary of required and recommended fixes

1. **Required if LLMs were used:** add a first-page footnote disclosing the LLM use (FAQ requirement).
2. **Recommended:** trim at least 3 lines of main text so the main body is at most 12 pages before References.
3. **Recommended:** change "in this revision" (sections/10_appendix.tex:202) to neutral wording, and consider de-"revision"-ing the script names.
4. **Cosmetic:** remove the upscaling `\resizebox` on Table 3 (12.5 pt text) and Table 2.
5. **Camera-ready only:** real `\author`, `\month`/`\year`, `\openreview`, the `[accepted]` option, and Acknowledgments.
