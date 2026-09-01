# Windows convenience wrapper. Same steps as the Makefile.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "`n[1/5] Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "`n[2/5] Fetching SEC filings (~165 MB, cached after the first run)..." -ForegroundColor Cyan
python -m scripts.fetch_sec

Write-Host "`n[3/5] Building corpus, portfolio and eval set..." -ForegroundColor Cyan
python -m scripts.build_corpus
python -m scripts.build_portfolio
python -m scripts.build_eval_set

Write-Host "`n[4/5] Indexing into Qdrant (embeds ~1,350 chunks; several minutes)..." -ForegroundColor Cyan
python -m scripts.index_corpus

Write-Host "`n[5/5] Running the evaluation..." -ForegroundColor Cyan
python -m scripts.evaluate

Write-Host "`nDone. Start the API with:  python -m uvicorn app.main:app --reload" -ForegroundColor Green
