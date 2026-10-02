$ErrorActionPreference = 'Stop'
$projectRoot = (Get-Location).Path
$edaMarkdownFiles = @('EDA/10.02_001_데이터_수록내용_선별.md','EDA/10.02_002_EDA_새원고.md','EDA/README.md')
foreach ($markdownFile in $edaMarkdownFiles) {
    $text = Get-Content -Encoding UTF8 -Raw -LiteralPath $markdownFile
    $text = [regex]::Replace($text,'\[([^\]]+\.py)\]\(scripts/([^\)]+\.py)\)',{
        param($match)
        $scriptName = $match.Groups[2].Value
        $absolutePath = (Join-Path $projectRoot "EDA/scripts/$scriptName").Replace('\','/')
        return "[EDA/scripts/$scriptName](<$absolutePath>)"
    })
    Set-Content -Encoding UTF8 -NoNewline -LiteralPath $markdownFile -Value $text
}
$backupBase = Join-Path $projectRoot '백업/EDA_검수/2026-10-02'
$reviewImages = @('EDA/figures/daily_repetition/simple_contact_50.png','EDA/figures/daily_repetition/monthly_mean_heatmap_contact_50.png','EDA/figures/qa/manuscript_contact_01_50.png','EDA/figures/qa/manuscript_contact_02_50.png','EDA/figures/qa/manuscript_contact_03_50.png')
$moveRecords = @()
foreach ($sourceRelative in $reviewImages) {
    $sourcePath = (Resolve-Path -LiteralPath $sourceRelative).Path
    $targetPath = [IO.Path]::GetFullPath((Join-Path $backupBase ([IO.Path]::GetFileName($sourcePath))))
    foreach ($checkedPath in @($sourcePath,$targetPath)) {
        if (-not $checkedPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $checkedPath" }
    }
    if (Test-Path -LiteralPath $targetPath) { throw "Destination exists: $targetPath" }
    $moveRecords += [pscustomobject]@{source=$sourcePath;target=$targetPath;sha256=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash}
}
$oldReport = (Resolve-Path -LiteralPath 'report').Path
$backupReport = [IO.Path]::GetFullPath((Join-Path (Resolve-Path -LiteralPath '백업').Path 'report'))
foreach ($checkedPath in @($oldReport,$backupReport)) {
    if (-not $checkedPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $checkedPath" }
}
if (Test-Path -LiteralPath $backupReport) { throw '백업/report already exists' }
$reportFiles = Get-ChildItem -LiteralPath $oldReport -Recurse -File -Force
$reportManifest = @($reportFiles | ForEach-Object { [pscustomobject]@{relative=$_.FullName.Substring($oldReport.Length+1);sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })
New-Item -ItemType Directory -Path $backupBase -Force | Out-Null
foreach ($record in $moveRecords) {
    Move-Item -LiteralPath $record.source -Destination $record.target
    if ((Get-FileHash -LiteralPath $record.target -Algorithm SHA256).Hash -ne $record.sha256) { throw 'Review image hash mismatch' }
}
$qaPath = (Resolve-Path -LiteralPath 'EDA/figures/qa').Path
if (-not $qaPath.StartsWith((Join-Path $projectRoot 'EDA') + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected QA path' }
if (@(Get-ChildItem -LiteralPath $qaPath -Force).Count -eq 0) { Remove-Item -LiteralPath $qaPath }
Move-Item -LiteralPath $oldReport -Destination $backupReport
foreach ($record in $reportManifest) {
    $movedPath = Join-Path $backupReport $record.relative
    if ((Get-FileHash -LiteralPath $movedPath -Algorithm SHA256).Hash -ne $record.sha256) { throw "Report file hash mismatch: $movedPath" }
}
$reportManifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $backupBase 'report_archive_manifest.json')
$moveRecords | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $backupBase 'review_image_manifest.json')
Write-Output "Archived five QA images and $($reportManifest.Count) report files; hash verification passed."
