# Extracts Word .docx spec into plain text for easy searching/implementation.
# This is used to implement the exact FSM + gesture thresholds from the thesis spec.

param(
  [string]$InputDocx = "PRACA INZYNIERSKA.docx",
  [string]$OutputTxt = "docs/SPEC_EXTRACTED.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputDocx)) {
  throw "Input file not found: $InputDocx"
}

$tmp = ".spec_tmp"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $tmp | Out-Null

$zipPath = Join-Path $tmp "spec.zip"
Copy-Item $InputDocx $zipPath -Force

Expand-Archive $zipPath -DestinationPath $tmp -Force

$xmlPath = Join-Path $tmp "word/document.xml"
if (-not (Test-Path $xmlPath)) {
  throw "document.xml not found inside docx: $xmlPath"
}

[xml]$xml = Get-Content -Raw $xmlPath
$ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
$ns.AddNamespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")

# Preserve paragraph boundaries: each w:p becomes one line in the output.
$paras = $xml.SelectNodes("//w:p", $ns)
$lines = foreach ($p in $paras) {
  ($p.SelectNodes(".//w:t", $ns) | ForEach-Object { $_.InnerText }) -join ""
}

$outDir = Split-Path -Parent $OutputTxt
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force $outDir | Out-Null
}

$lines | Out-File -FilePath $OutputTxt -Encoding UTF8
Write-Host "Extracted to $OutputTxt"

