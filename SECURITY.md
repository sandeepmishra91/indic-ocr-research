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
