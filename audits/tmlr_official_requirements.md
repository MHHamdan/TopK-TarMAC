# TMLR official requirements (fetched 2026-10-03)

Sources fetched (HTTP 200, page text read via curl; footer on every page reads "© TMLR 2026"):

| Key | URL |
|---|---|
| HOME | https://jmlr.org/tmlr/ (index.html) |
| AUTH | https://jmlr.org/tmlr/author-guide.html |
| SUBM | https://jmlr.org/tmlr/submissions.html |
| POL | https://jmlr.org/tmlr/editorial-policies.html |
| ACC | https://jmlr.org/tmlr/acceptance-criteria.html |
| REV | https://jmlr.org/tmlr/reviewer-guide.html |
| AE | https://jmlr.org/tmlr/ae-guide.html |
| FAQ | https://jmlr.org/tmlr/faq.html |
| ETH | https://jmlr.org/tmlr/ethics.html |
| BPDF | https://tmlr-beyond-pdf.org/about (redirect target of https://tmlr-beyond-pdf.org) |
| STY | https://github.com/JmlrOrg/tmlr-style-file (last commit 7bf90ef, 2023-06-30, "Removed duplicated line in tmlr.sty") |

There is no separate "LLM policy" page or separate "certifications" page. The LLM policy is a section of POL plus a FAQ entry. Certifications are a section of POL and REV.

## 1. Scope
- POL: "TMLR's objective is to publish original papers that contribute to the understanding of the computational and mathematical principles that enable intelligence through learning". Listed types include "experimental and/or theoretical studies yielding new insight into the design and behavior of learning in intelligent systems" and "accounts of applications of existing techniques that shed light on the strengths and weaknesses of the methods".
- HOME: "TMLR emphasizes technical correctness over subjective significance".
- FAQ: "As of September 1, 2026, TMLR no longer considers survey papers for review." This is new and does not affect this manuscript.
- FAQ: "TMLR does not have a preference" between methodological, theoretical and application work.

## 2. Acceptance criteria
- POL / ACC: the two questions are "Are the claims made in the submission supported by accurate, convincing and clear evidence?" and "Would at least some individuals in TMLR's audience be interested in knowing the findings of this paper?"
- POL: "Papers should be accepted if they meet the criteria, even if the contribution or significance of the work is modest." Papers that should not be accepted include those "that make bold statements unsupported by empirical or rigorous evidence, papers that aren't clearly written, papers that incorrectly claim novelty over existing published work".
- ACC: "novelty of the studied method is not a necessary criterion for acceptance in TMLR". The interest criterion "should not be used as a reason to reject work that isn't considered 'significant' or 'impactful'".

## 3. Page length
- AUTH: "Submissions may be any length, but a paper's length should be justified by its content and papers that are unusually long (not counting any Appendices) are likely to result in reviewing delays." **There is no hard page limit.**
- The 12-page threshold sets the review timeline. It is not a limit on submissions:
  - REV: "reviews must be submitted within 2 weeks of their assignment by the AE for submissions up to 12 pages (of main content, before references), and 4 weeks for submissions over 12 pages."
  - FAQ: "We ask for reviews to be submitted within 2 weeks for submissions of up to 12 pages (main content, before references). For submissions over 12 pages, we ask for reviews to be submitted within 4 weeks."
  - FAQ: "manuscripts whose main body exceeds 12 pages are subject to longer timescales"
  - AE: "if the submission is longer than 12 pages (excluding any appendix), we ask you to permit a longer reviewing time frame."
- **What counts toward the 12 pages:** main content before the references. References and appendices do not count. No page states whether the title block, the Broader Impact Statement or any other unnumbered statement placed before the references is excluded, so assume it counts.
- AE: standard submissions are of type "long" or "regular" on OpenReview. No official page defines the threshold between them. Presumably it is the 12-page threshold, but this is not stated.

## 4. Appendices and supplementary material
- AUTH: "The submission PDF may include an Appendix, after the references (for proofs, derivations or complimentary results). Looking at the Appendix is at the discretion of the reviewers."
- AUTH: "Authors may submit up to 100MB of supplementary material, such data, source code or illustrative videos; all supplementary materials must be in PDF or ZIP format." Also: "Like submissions, supplementary material must be anonymized."
- FAQ: "Appendices can be included in the main pdf. The supplementary zip file should contain additional material such as videos or code."

## 5. Double-blind and anonymity
- AUTH: "TMLR uses a double blind review process and submissions must be anonymized."
- POL: "TMLR uses an open-reviewing, double-blind process." Once reviewers are assigned, "the paper will become public".
- AE: "Clear cases of desk rejection include submissions that are not anonymized, submissions that do not use the unmodified TMLR stylefile and submissions that clearly overlap with work already published in proceedings (or currently under review for publication)."
- Official example main.tex comment: "Non-anonymous submissions will be rejected without review."
- AUTH: preprints are allowed, "though double blind of the TMLR submission itself must be maintained by not linking to another version that includes the authors' names."
- FAQ on resubmissions: "Only the action editor (not the reviewers) will see the previous submission URL".

## 6. OpenReview process
- SUBM: submissions go through https://openreview.net/group?id=TMLR. After submitting you "receive an email ... asking you to recommend potential Action Editors".
- AUTH: "All authors must have complete and active OpenReview profiles, including information such as affiliations, conflicts of interest, and publication history." Authors also provide information on "human subjects reporting (IRB), funding, competing interests". This information is not shown to reviewers.
- POL phases:
  - Submission, in anonymized form.
  - AE assigned "within a week". The AE may desk-reject, including for "format violations (contingent on Editor-in-Chief approval)". Otherwise there are "at least three qualified reviewers".
  - Rebuttal and discussion. The recommendation is to wait for all 3 reviews before posting a revised PDF.
  - Decision: "Accept as is", "Accept with minor revision" or "Reject".
- POL: "The exact set of authors must be listed on OpenReview ... No authors may be added or removed after submission."
- POL: submission quotas follow the "Generalized Harmonic Quota Rule, with parameters N_1 = 2 and N_9 = 9". Desk-rejected submissions also use quota budget.
- FAQ: the discussion phase lasts "at least 2 weeks and up to 4 weeks". The target is a decision "approximately 9 weeks after submission".
- POL: a rejected paper can be resubmitted only as "a new submission and a link provided to the previously rejected submission as well as a description of the changes made since."

## 7. LLM use and disclosure
- POL ("LLM use policy for authors"): "LLMs may be used as general-purpose assistive tools. Whichever tools are used, authors are fully responsible for content on which they are listed as (co-) authors. This includes, but is not limited to, content generated by LLMs that could be construed as plagiarism or scientific misconduct (e.g., fabrication of facts). Low-quality contributions ... that appear to be largely LLM-generated will be closely examined ... LLMs are not eligible for authorship."
- **Disclosure requirement (FAQ, "Can I use ChatGPT or other large language model systems to assist in writing my TMLR submission?"):** "Additionally, transparency is always good, so we also require that authors mention explicitly in their submission that they have used this tool, as a footnote on the first page."
  - The same answer says: "our expectation is that the ideas, claims and results found in submissions are human-sourced, not produced by such tools."
  - **Where:** a footnote on the first page of the submission.
  - **What it must say:** an explicit statement that the LLM tool was used. No required wording or level of detail is specified. Naming the tool and describing the kind of use is the reasonable reading of "this tool", but neither is stated.
- POL does not repeat the footnote requirement. Only the FAQ states it, and the FAQ also points to POL for the author policy. Treat the footnote as required.

## 8. Broader impact statement
- AUTH: "If their work carries a significant risk of harm, authors are required to include a Statement of Broader Impact. The statement should include a concise, concrete description of potential positive and negative societal impact, as well as a discussion of possible mitigations."
- POL: "Submissions which describe work with the potential for harm should clearly describe the risks and mitigations in an (otherwise optional) Broader Impact Statement."
- ETH: "Submissions to TMLR are expected, when applicable, to include a discussion about potential negative societal impacts ... Whenever these are identified, submissions should also include a discussion about how these risks can be mitigated." ETH also says TMLR "reserves the right to reject submissions that have violated key ethical principles."
- REV: the review form has a "Broader impact concerns" field.
- FAQ (AE): an AE can require a broader impact statement "for the camera-ready version".
- **Summary:** the statement is required only for work with significant risk of harm and is optional otherwise. The official template includes a `\subsubsection*{Broader Impact Statement}` before the bibliography.

## 9. Reproducibility
- AUTH: "We encourage authors to, whenever possible, upload materials that improve reproducibility. Specifically, it is suggested that authors submit supplementary material including data or code towards this end." This is encouraged, not required.
- POL: a "Reproducibility Certification" exists for papers "whose primary purpose is reproduction of other published work".
- POL: after acceptance, "The author is encouraged to submit additional materials ... such as a video presentation or code."
- No mandatory reproducibility checklist or reproducibility statement was found on any official page.

## 10. Dual submission, originality, preprints and workshops
- POL: "TMLR only accepts original contributions that don't reuse the authors' own prior work. In particular, we do not accept submissions that are expanded versions of conference papers. There should not be any reuse of written text, figures or results between the submitted paper and any paper which has been published, accepted for publication, or submitted in parallel at another archival, peer-reviewed venue."
- POL: "It is acceptable for a submission to overlap with the author's previous work if it was shared at venues or tracks that are publicly declared, in writing, to be non-archival, such as workshops, or on preprint servers such as arXiv and bioRxiv."
- FAQ: you can submit "even if a preprint of your work already is available online somewhere (as long as it's not in official proceedings for some other publishing venue)."
- FAQ: if you post to arXiv using the TMLR template, use "the 'preprint' option of the stylefile".
- FAQ: a thesis is treated like a preprint.

## 11. Template and citation style
- AUTH: "Submissions must be PDF files generated using the TMLR LaTeX stylefile and template" (link: github.com/JmlrOrg/tmlr-style-file).
- SUBM: "The Author Guidelines page contains a LaTeX stylefile and template; these are mandatory. Any changes to the stylefile or template that alters the formatting, font, or layout of the manuscript may result in rejection without review."
- AE: desk rejection applies to "submissions that do not use the unmodified TMLR stylefile". The stylefile requirement "only applies to standard PDF submissions, *not* Beyond PDF submissions".
- STY `tmlr.sty`:
  - Loads natbib with `\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}`, i.e. author-year citations in round brackets.
  - Example uses `\bibliographystyle{tmlr}`.
  - Page is US Letter, 8.5in x 11in, with `\textwidth 6.5 true in` and `\textheight 9.0 true in`.
  - Fonts are T1 and lmodern. The class is `article` at 10pt.
- STY options:
  - None: review mode. Header "Under review as submission to TMLR", authors replaced by "Anonymous authors / Paper under double-blind review".
  - `accepted`: header "Published in Transactions on Machine Learning Research (\month/\year)" plus "Reviewed on OpenReview: \openreview".
  - `preprint`: de-anonymized, empty header.
- Camera-ready (AE): it must comply with the "TMLR stylefile, with all author information inserted in the manuscript as well as the link to the OpenReview page for the submission."
- POL: "Post-Publication Changes" sections go after Acknowledgments and before References. This applies only after publication.
- Licence (AUTH): "All TMLR submissions, from the time of submission to final publication, are licensed under CC BY 4.0."

## 12. Beyond PDF (interactive option)
- HOME: "TMLR Beyond PDF is a new, HTML-based submission format for TMLR, that supports interactive figures and videos, along with the usual LaTeX and images." It was announced 2025-11-25.
- BPDF: it uses "Markdown/HTML documents that support interactive figures, videos, or other rich media". To submit, select the "Beyond PDF submission (pageless, webpage-style content)" type on OpenReview and upload a zipped folder.
- BPDF: detailed instructions are at https://tmlr-beyond-pdf.org/submission-instructions. That page was not fetched.
- The same editorial policies apply. The stylefile requirement does not.

## 13. Certifications (POL; also REV)
- Outstanding, Featured, Reproducibility and Survey certifications.
- J2C Certification: "Awarded to publications based on responses by the AE and reviewers, to the question of whether the work would be appropriate for the joint NeurIPS/ICLR/ICML Journal-to-Conference Track".
- Expert Certification: for papers by Expert Reviewers.
- Accepted papers "will be automatically considered". Authors do not apply.

## Rules looked for and not found
- No hard page limit, and no rule defining "long" versus "regular" submission types.
- No mandated wording for the LLM footnote.
- No mandatory reproducibility checklist or statement.
- No font-size minimum for figures or tables beyond "unmodified stylefile". No explicit ban on `\resizebox` or `\small`. SUBM only forbids changes that alter "formatting, font, or layout".
- Not fetched: https://tmlr-beyond-pdf.org/submission-instructions.
