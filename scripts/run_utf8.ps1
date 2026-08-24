param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('1', '2', '3', '4', 'pii', 'json')]
    [string]$Step
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'venv\Scripts\python.exe'
$sourceRoot = Join-Path $projectRoot 'src'
$evidenceRoot = Join-Path $projectRoot 'evidence'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $evidenceRoot)) {
    New-Item -ItemType Directory -Path $evidenceRoot | Out-Null
}

function Invoke-LabStep {
    param(
        [string[]]$Arguments,
        [string]$LogPath = ''
    )

    $captured = @()
    # Windows PowerShell 5.1 wraps native stderr as ErrorRecord. With the
    # script-wide Stop policy, the first traceback line would terminate this
    # pipeline and hide the actual Python exception. Temporarily continue so
    # the complete output is displayed/captured, then fail from the real exit
    # code below.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python @Arguments 2>&1 | Tee-Object -Variable captured
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($LogPath) {
        $lines = [string[]]@($captured | ForEach-Object { $_.ToString() })
        $logText = ($lines -join "`n") + "`n"
        [IO.File]::WriteAllText($LogPath, $logText, $utf8NoBom)
        Write-Host "UTF-8 log saved: $LogPath"
    }

    if ($nativeExitCode -ne 0) {
        throw "Lab step failed with exit code $nativeExitCode"
    }
}

Push-Location $projectRoot
try {
    switch ($Step) {
        '1' {
            Invoke-LabStep -Arguments @((Join-Path $sourceRoot '01_langsmith_rag_pipeline.py'))
        }
        '2' {
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '02_prompt_hub_ab_routing.py')) `
                -LogPath (Join-Path $evidenceRoot '02_ab_routing_log.txt')
        }
        '3' {
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '03_ragas_evaluation.py')) `
                -LogPath (Join-Path $evidenceRoot '03_ragas_console_log.txt')
        }
        '4' {
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '04_guardrails_validator.py'), '--demo', 'pii') `
                -LogPath (Join-Path $evidenceRoot '04_pii_demo_log.txt')
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '04_guardrails_validator.py'), '--demo', 'json') `
                -LogPath (Join-Path $evidenceRoot '04_json_demo_log.txt')
        }
        'pii' {
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '04_guardrails_validator.py'), '--demo', 'pii') `
                -LogPath (Join-Path $evidenceRoot '04_pii_demo_log.txt')
        }
        'json' {
            Invoke-LabStep `
                -Arguments @((Join-Path $sourceRoot '04_guardrails_validator.py'), '--demo', 'json') `
                -LogPath (Join-Path $evidenceRoot '04_json_demo_log.txt')
        }
    }
}
finally {
    Pop-Location
}
