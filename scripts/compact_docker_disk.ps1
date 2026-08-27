# Reclaim the space Docker has already freed *inside* its virtual disk.
#
#   Run in an ELEVATED PowerShell (Right-click > Run as Administrator):
#     powershell -ExecutionPolicy Bypass -File D:\repos\hvym-img-tools\scripts\compact_docker_disk.ps1
#
# Why this is needed at all: `docker builder prune` frees space inside the VHDX,
# but the file on C: never shrinks on its own. Windows only returns those blocks
# when the virtual disk is compacted, and `diskpart compact vdisk` requires
# elevation -- which is the only reason this is a separate script.
#
# Compacting is NON-DESTRUCTIVE: it removes unused blocks only. Images,
# containers and volumes are untouched. It is safe to re-run.
#
# WSL's `--set-sparse` alternative is deliberately not used here: Microsoft
# currently gates it behind --allow-unsafe over data-corruption risk.

$ErrorActionPreference = 'Stop'

$vhdx = "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This needs an elevated PowerShell (Run as Administrator)." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $vhdx)) {
    Write-Host "Not found: $vhdx" -ForegroundColor Red
    Write-Host "Docker may store its disk elsewhere; check Settings > Resources > Advanced."
    exit 1
}

function Free-GB { (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB }

$sizeBefore = (Get-Item $vhdx).Length / 1GB
$freeBefore = Free-GB
Write-Host ("disk image : {0:N1} GB" -f $sizeBefore)
Write-Host ("C: free    : {0:N1} GB" -f $freeBefore)

Write-Host "`nStopping Docker Desktop and WSL..." -ForegroundColor Cyan
Get-Process 'Docker Desktop', 'com.docker.backend', 'com.docker.build' -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
wsl --shutdown
Start-Sleep -Seconds 5

# The disk must be detached before it can be compacted; attaching read-only
# guarantees nothing can be written to it while we work.
Write-Host "Compacting (this can take several minutes)..." -ForegroundColor Cyan
$script = @"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
$f = Join-Path $env:TEMP 'compact_docker_disk.txt'
Set-Content -Path $f -Value $script -Encoding ascii
try {
    diskpart /s $f | Select-Object -Last 15
} finally {
    Remove-Item $f -ErrorAction SilentlyContinue
}

$sizeAfter = (Get-Item $vhdx).Length / 1GB
$freeAfter = Free-GB
Write-Host ""
Write-Host ("disk image : {0:N1} GB -> {1:N1} GB  (freed {2:N1} GB)" -f $sizeBefore, $sizeAfter, ($sizeBefore - $sizeAfter)) -ForegroundColor Green
Write-Host ("C: free    : {0:N1} GB -> {1:N1} GB" -f $freeBefore, $freeAfter) -ForegroundColor Green
Write-Host "`nStart Docker Desktop again when you need it."
