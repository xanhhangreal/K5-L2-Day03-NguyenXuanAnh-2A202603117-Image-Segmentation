param([string[]]$Tasks, [switch]$CheckOnly, [switch]$Rebuild, [switch]$Improve)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$unix = (& wsl -d Ubuntu -- wslpath -a ($repo -replace '\\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve WSL repository path' }
& wsl -d Ubuntu -- bash /home/ad/sam-setup/start.sh
if ($LASTEXITCODE -ne 0) { throw 'CVAT/SAM startup failed' }
& wsl -d Ubuntu -- docker image inspect cvat.lab3.auto:latest --format '{{.Id}}'
$imageMissing = $LASTEXITCODE -ne 0
if ($imageMissing -or $Rebuild) {
    & wsl -d Ubuntu -- docker build -t cvat.lab3.auto:latest "$unix/tools/lab3_auto"
    if ($LASTEXITCODE -ne 0) { throw 'Runner image build failed' }
} else {
    Write-Host 'Reusing the installed runner image (no downloads). Use -Rebuild only to rebuild it.'
}
if ($Improve -and ($Tasks -or $CheckOnly)) { throw '-Improve processes Medium/Hard and cannot be combined with -Tasks or -CheckOnly.' }
$image = 'cvat.lab3.auto:latest'
$script = 'tools/lab3_auto/run.py'
if ($Improve) {
    $image = 'cvat.lab3.improve:latest'
    $script = 'tools/lab3_auto/improve.py'
    & wsl -d Ubuntu -- docker image inspect $image --format '{{.Id}}'
    if ($LASTEXITCODE -ne 0 -or $Rebuild) {
        & wsl -d Ubuntu -- docker build -f "$unix/tools/lab3_auto/Dockerfile.improve" -t $image "$unix/tools/lab3_auto"
        if ($LASTEXITCODE -ne 0) { throw 'Improvement image build failed' }
    }
}
$dockerArgs = @('docker', 'run', '--rm', '--name', 'lab3-auto-runner', '--network', 'cvat_cvat',
    '-v', "${unix}:/workspace", '-v', '/home/ad/sam-setup/auto-cache:/cache',
    '-v', '/var/run/docker.sock:/var/run/docker.sock', $image,
    'python', $script)
if ($Tasks) { $dockerArgs += '--tasks'; $dockerArgs += $Tasks }
if ($CheckOnly) { $dockerArgs += '--check-only' }
if ($Improve) { $dockerArgs += '--promote' }
& wsl -d Ubuntu -- @dockerArgs
if ($LASTEXITCODE -ne 0) { throw 'Some tasks failed. See AUTOMATION_RUN_REPORT.md; rerun to resume.' }
