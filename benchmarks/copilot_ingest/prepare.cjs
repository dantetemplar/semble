// Optional adapter around the same WASM utilities used by VS Code Copilot.
// No HTTP, credentials, or workspace reads: public document bytes arrive on stdin.
const crypto = require('node:crypto');
const fs = require('node:fs');
const ingest = require('@github/blackbird-external-ingest-utils');

ingest.setupPanicHooks();
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const toBase64 = (bytes) => Buffer.from(bytes).toString('base64');
let result;
if (request.action === 'symbols') {
  const { start, end } = request.range;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)
      || start < 0 || end < start || end - start > 100000) {
    throw new Error('Invalid coded-symbol range');
  }
  const shas = request.doc_ids.map((value) => Uint8Array.from(Buffer.from(value, 'base64')));
  result = { coded_symbols: ingest.createCodedSymbols(shas, start, end).map(toBase64) };
} else if (request.action === 'prepare') {
  const filter = new ingest.IngestFilter();
  const geoFilter = new ingest.GeoFilter();
  const checkpoint = crypto.createHash('sha256');
  const documents = [];
  const skipped = [];
  const shas = [];
  for (const file of request.files) {
    const bytes = Buffer.from(file.content, 'base64');
    const contents = new ingest.DocumentContents(bytes);
    const verdict = ingest.canIngestDocument(filter, file.path, contents);
    const reason = verdict.failureReason;
    verdict.free();
    if (reason !== undefined) {
      skipped.push({ path: file.path, reason });
      contents.free();
      continue;
    }
    const sha = ingest.getDocSha(file.path, contents);
    contents.free();
    if (!(sha instanceof Uint8Array) || !sha.length) {
      throw new Error(`No document SHA for ${file.path}`);
    }
    geoFilter.push(sha);
    shas.push(sha);
    documents.push({ path: file.path, doc_id: toBase64(sha) });
    checkpoint.update(`file:///semble-bench/${encodeURIComponent(request.repo)}/${encodeURI(file.path)}`);
    checkpoint.update(sha);
  }
  result = {
    documents,
    skipped,
    checkpoint: checkpoint.digest('base64'),
    geo_filter: toBase64(geoFilter.toBytes()),
    coded_symbols: ingest.createCodedSymbols(shas, 0, 1).map(toBase64),
    cache_version: ingest.cacheVersion(),
  };
  geoFilter.free();
  filter.free();
} else {
  throw new Error('Unknown helper action');
}
process.stdout.write(JSON.stringify(result));
