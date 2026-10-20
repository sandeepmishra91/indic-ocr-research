# Dataset contract

Each nonblank JSONL line is an object with string fields:

```json
{"sample_id":"doc01-line01","image_path":"lines/01.png","transcription":"नमस्ते","language":"hin","script":"Devanagari","document_id":"doc01","split":"train","source_url":"","license":"your documented permission","annotation_method":"human-double-reviewed","task":"line"}
```

This is a schema example, not a shipped label or actual document. `task` is `line`, `block`
or `page`; `split` is `train`, `validation` or `test`. All keys are required;
only transcription and source_url may be empty. Use a private source_url only in
private manifests. Blank transcription is valid for negative examples, but warn
on it and never derive a vocabulary solely from blank labels.

Obtain documents through authorized channels, record provenance and redistribution
permission, render pages, crop lines for CRNN, and have annotators transcribe visible
text. Preserve the source document ID across all crops, revisions and pages. Verify
Hindi/Malayalam text with qualified readers and record disagreement resolution.
Do not use OCR output as human ground truth. Maintain a separate held-out test set.
Redact personal information and confidential fields before any public contribution.
Government origin does not by itself establish redistribution permission.

Validation reports counts by language/split/task, duplicate sample IDs, document
leakage, image byte duplicates, conflicting labels, missing and unreadable images.
Byte hashes do not detect perceptually identical re-encoded images; document IDs
must accurately group these. NFC normalization defaults on during ingestion, is
configurable in validation, and is mandatory for the CRNN vocabulary. Original
benchmark labels are retained for raw metrics. Manifest splitting is deterministic
by seed and document, but not stratified by language. Always inspect small splits.
Optional Hugging Face imports require explicit licensed, pinned recipes; see
[hugging-face.md](hugging-face.md). No general crawling or unclear-license download is implemented. Raw documents and checkpoints
are ignored by Git. Keep private data outside the public repository when possible.
