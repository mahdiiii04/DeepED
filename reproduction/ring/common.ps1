$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:CUDA_VISIBLE_DEVICES = "-1"
$CFG = "../configs/ring"
if (-not $SEEDS) { $SEEDS = "0,1,2,3,4" }

function sw {
    param([string]$name, [string]$val, [Parameter(ValueFromRemainingArguments=$true)]$rest)
    $a = @("./scripts/train.py", "--config-path", $CFG, "--config-name", "deeped_safe",
           "--multirun", "experiment_name=RingSweep_${name}_${val}") + $rest + @("seed=$SEEDS")
    python @a
}
