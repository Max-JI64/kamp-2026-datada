$ErrorActionPreference = 'Stop'
$projectRoot = (Get-Location).Path
$selectedMoves = @(
    @('report/10.02_001_데이터_수록내용_선별.md','EDA/10.02_001_데이터_수록내용_선별.md'),
    @('report/10.02_002_EDA_새원고.md','EDA/10.02_002_EDA_새원고.md')
)
foreach ($scriptName in @('eda_daily_pattern.py','eda_variable_relations.py','eda_figures.py','eda_daily_repetition.py','plot_daily_repetition_simple.py','verify_monthly_power_heatmap.py','plot_monthly_power_heatmap.py')) {
    $selectedMoves += ,@("report/EDA/scripts/$scriptName","EDA/scripts/$scriptName")
}
$selectedMoves += ,@('report/EDA/scripts/eda_analysis.py','EDA/scripts/data_overview.py')
foreach ($figureName in @('restart_01_daily_pattern_clean.png','restart_02_variable_relations.png','restart_02_monthly_relations.png')) {
    $selectedMoves += ,@("report/figures/eda/$figureName","EDA/figures/$figureName")
}
foreach ($figureName in @('04_daily_pattern_simple.png','07_monthly_power_mean_heatmap.png')) {
    $selectedMoves += ,@("report/EDA/figures/daily_repetition/$figureName","EDA/figures/daily_repetition/$figureName")
}
foreach ($tableName in @('hourly_overview.csv','daily_pattern_restart.csv','daily_pattern_restart_manifest.json','restart_relations_distribution.csv','restart_relations_sensitivity.csv','restart_relations_bins.csv','restart_relations_spearman.csv','restart_relations_monthly.csv','restart_relations_august_context.csv','restart_relations_august_week.csv','restart_relations_manifest.json')) {
    $selectedMoves += ,@("report/EDA/tables/$tableName","EDA/tables/$tableName")
}
foreach ($tableName in @('daily_values.csv','hourly_matrix.csv','hourly_distribution.csv','group_summary.csv','monthly_summary.csv','sign_combinations.csv','profile_sensitivity.csv','production_context.csv','summary.json','plan.json','simple_figure_manifest.json','monthly_hourly_distribution.csv','independent_monthly_cells.csv','july_august_levels.csv','july_august_hour_examples.csv','july_august_production_composition.csv','independent_heatmap_verification.json','monthly_mean_heatmap_manifest.json')) {
    $selectedMoves += ,@("report/EDA/tables/daily_repetition/$tableName","EDA/tables/daily_repetition/$tableName")
}
$records = @()
foreach ($pair in $selectedMoves) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $projectRoot $pair[0]))
    $targetPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $pair[1]))
    foreach ($checkedPath in @($sourcePath,$targetPath)) {
        if (-not $checkedPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw "Outside project: $checkedPath" }
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw "Missing source: $sourcePath" }
    if (Test-Path -LiteralPath $targetPath) { throw "Destination already exists: $targetPath" }
    $records += [pscustomobject]@{ source=$pair[0]; target=$pair[1]; original_sha256=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() }
}
foreach ($record in $records) {
    $sourcePath = Join-Path $projectRoot $record.source
    $targetPath = Join-Path $projectRoot $record.target
    $targetDirectory = Split-Path -Parent $targetPath
    if (-not (Test-Path -LiteralPath $targetDirectory)) { New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null }
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.original_sha256) { throw "Hash changed during move: $targetPath" }
}
$records | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath 'EDA/tables/migration_manifest.json'
Write-Output "Moved and hash-checked $($records.Count) selected files."
