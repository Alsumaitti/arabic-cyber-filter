# LLM-as-Judge Protocol — Arabic Cybersecurity Corpus Verification

Judge: Claude (Fable 5), judging directly in-session. One pass per record, blind to
desired outcome. Judgments are recorded per batch in `evaluation/judgments/`.

## Question

> Is this document **substantively about cybersecurity**?

"Cybersecurity" covers: cyber attacks, malware (viruses, trojans, ransomware, spyware…),
hacking/hackers, data breaches, vulnerabilities & exploits, phishing/scams/social
engineering, network/system security, encryption & privacy protection, digital forensics,
cybercrime & cyber law, cyber warfare, security products (antivirus, firewalls, VPNs),
security awareness/training, account/device protection guides.

## Labels

| Label | Rule |
|---|---|
| `cyber` | Cybersecurity is a main topic of the document, or a substantial section of it (a security news story, a malware removal guide, an infosec explainer, cyber-attack coverage, a how-to-protect-yourself article, security product review…) |
| `borderline` | Cybersecurity is genuinely present but tangential/minor: one paragraph in a general news roundup, a passing mention of hacking in a tech/politics article, general IT content with light security elements |
| `not_cyber` | The document is not about cybersecurity. Keyword hit is a false positive: metaphorical use, different sense (e.g. physical security, military "الحرب الإلكترونية" as electronic warfare in a purely military-hardware sense is still cyber-adjacent → judge on substance), spam/aggregator pages whose text isn't security content |

## Per-record output fields

- `judge_label`: cyber | borderline | not_cyber
- `judge_confidence`: high | medium | low (low = truncated view left real doubt)
- `judge_justification`: one English sentence, citing the concrete evidence in the text
- `judge_summary`: one English sentence stating what the document is about

## Metric conventions

- **Strict precision**: only `cyber` counts as true positive.
- **Lenient precision**: `cyber` + `borderline` count as true positive.
- Precision is estimated per stratum (1 / 2 / 3–5 / 6+ distinct keyword hits) and
  combined with corpus-frequency weights (9062 / 2209 / 1610 / 464 of 13,345).
- Recall is estimated by pooling: a broad "safety net" regex over the ~986k rejected
  records defines a candidate pool; a random sample of the pool and a random sample of
  the remainder are judged; false-negative mass is extrapolated from both rates.

## Judging discipline

- Judge only from the provided text view (beginning + window around first keyword hit).
- Do not be biased by `keywords_found` — the keyword list says why the filter kept it,
  not whether the document is actually about security.
- When the view is insufficient to decide between borderline and cyber, prefer the
  lower label and set confidence to low.
