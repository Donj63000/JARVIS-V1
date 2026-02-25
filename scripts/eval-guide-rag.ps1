param(
    [string]$Model = "qwen2.5-coder:14b",
    [string]$Guide = "guide-production-rochias.txt",
    [int]$MaxQuestions = 10
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

$questions = @(
    "Quel signal est mentionne en cas de defaut dans la salle de controle ?",
    "Combien de cellules sont decrites pour le sechoir ?",
    "Dans quel ordre demarre-t-on la ligne de production ?",
    "Peut-on demarrer le tapis 2 sans demarrer le tapis 3 ?",
    "Ou trouve-t-on le tableau des temperatures ?",
    "Quelle section parle de la saisie de production ?",
    "Quelle information renseigne le numero de lot ?",
    "Qui gere les donnees de debit ?",
    "Le guide est-il un mode operatoire officiel ?",
    "Quelles actions sont disponibles pour les operateurs ?"
)

$citationRegex = '\[REF\s+\d+\]'

$subset = $questions | Select-Object -First ([math]::Max(1, [math]::Min($MaxQuestions, $questions.Count)))
$passed = 0
$total = $subset.Count

for ($i = 0; $i -lt $subset.Count; $i++) {
    $q = $subset[$i]
    Write-Host ""
    Write-Host "[$($i+1)/$total] $q"
    $output = cargo run --release -- --cli --model $Model --guide $Guide --source-limit 6 $q 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "KO (execution CLI en erreur)"
        continue
    }
    $output = $output.Trim()

    $hasCitation = $output -match $citationRegex
    $isNotFound = $output -match 'Information non trouvee dans le guide fourni\.'
    if ($hasCitation -or $isNotFound) {
        $passed++
        Write-Host "OK"
    }
    else {
        Write-Host "KO (pas de citation detectee)"
    }
}

$rate = [math]::Round(($passed * 100.0) / [math]::Max($total, 1), 1)
Write-Host ""
Write-Host "Resultat: $passed/$total ($rate%)"
