param(
    [int]$N = 131072,
    [string]$InputType = "uniform",
    [int]$Seed = 2026,
    [int]$Repeats = 5,
    [int[]]$Threads = @(1,2,4,6,8,12)
)

function Get-Field([string[]]$lines, [int]$idx) {
    $data = $lines | Where-Object { $_ -and -not $_.StartsWith("hardware_id") }
    if (-not $data) { throw "нет данных в выводе" }
    
    return [double](($data[0] -split ',')[$idx])
}

Write-Host "N=$N input=$InputType seed=$Seed"

$serLines = .\sort_cpu.exe --mode serial --n $N --input $InputType --seed $Seed --repeats $Repeats
$tser = Get-Field $serLines 9
Write-Host ("serial: {0:F3} ms" -f $tser)

$rows = @()
foreach ($p in $Threads) {
    $lines = .\sort_cpu.exe --mode omp --threads $p --n $N --input $InputType --seed $Seed --repeats $Repeats
    $t = Get-Field $lines 9
    $S = $tser / $t
    $E = $S / $p
    $rows += [pscustomobject]@{ Threads=$p; T_ms=[math]::Round($t,3); S=[math]::Round($S,3); E=[math]::Round($E,3) }
}

$rows | Format-Table -AutoSize