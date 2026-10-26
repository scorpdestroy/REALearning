$N = 131072
$tser = [double]((.\sort_cpu.exe --mode serial --n $N --input uniform --seed 2026 --repeats 5) -split ',' | Select-Object -Index 9)
"serial: $tser ms"
foreach ($p in 1,2,4,6,8,12) {
  $tmp = [double]((.\sort_cpu.exe --mode omp --threads $p --n $N --input uniform --seed 2026 --repeats 5) -split ',' | Select-Object -Index 9)
  $S = $tser / $tmp
  $E = $S / $p
  "{0,2} threads: T={1,8:F3} ms  S={2,5:F2}  E={3,5:F3}" -f $p, $tmp, $S, $E
}