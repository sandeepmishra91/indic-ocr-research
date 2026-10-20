# Security

Do not post credentials or private documents in public issues. Report sensitive
findings privately through the repository owner's GitHub security advisory channel
when enabled. No guaranteed response SLA is offered.

Treat input PDFs, images, YAML and checkpoints as untrusted. YAML uses safe loading;
checkpoints use torch weights_only loading but should still come from trusted sources.
Keep dependencies patched and process adversarial PDFs in an OS-isolated resource-limited
worker. Pixel limits do not bound every decoder/parser allocation. OCR subprocesses
use argument lists and a timeout. Do not expose this research CLI directly as a public
service without authentication, isolation, quotas and output sanitization.

Before committing, inspect staged content for tokens, personal information, private
source/data, .env/cloud credentials and large outputs. Generated reports include
transcriptions and local paths and must be reviewed before external sharing.

Hugging Face commands read only HF_TOKEN from the environment or explicitly selected
.env file. No token is recorded in import receipts or network error messages. Optional
imports require pinned revisions and declared open licenses; they never execute remote
dataset/model code. The opt-in hosted pilot uses a public dataset without secrets and
uploads summary metrics only. Keep detailed journals and third-party source data private
unless redistribution and privacy review explicitly permit sharing them.
