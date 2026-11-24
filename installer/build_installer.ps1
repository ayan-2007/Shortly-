$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Virtual environment Python not found at $Py"
}

Write-Host "[1/4] Generating app icon..." -ForegroundColor Cyan
& $Py -c "from PIL import Image; img = Image.open('shortsforge/static/shortly-icon.png').convert('RGBA'); img.save('installer/shortly.ico', format='ICO', sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])"

Write-Host "[2/4] Building standalone ShortlyStudio.exe with PyInstaller..." -ForegroundColor Cyan
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist\ShortlyStudio") { Remove-Item -Recurse -Force "dist\ShortlyStudio" }
& $Py -m PyInstaller --noconfirm --clean "installer\shortly.spec"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

Write-Host "[3/4] Staging config.yaml, .env.example, and bundled FFmpeg..." -ForegroundColor Cyan
$DistDir = Join-Path $Root "dist\ShortlyStudio"
Copy-Item (Join-Path $Root "config.yaml") (Join-Path $DistDir "config.yaml") -Force
Copy-Item (Join-Path $Root ".env.example") (Join-Path $DistDir ".env.example") -Force

$BinDir = Join-Path $DistDir "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$FfmpegPath = (& $Py -c "from shortsforge.ffmpeg_utils import get_ffmpeg_bin; print(get_ffmpeg_bin())").Trim()
$FfprobePath = (& $Py -c "from shortsforge.ffmpeg_utils import get_ffprobe_bin; print(get_ffprobe_bin())").Trim()
Copy-Item $FfmpegPath (Join-Path $BinDir "ffmpeg.exe") -Force
Copy-Item $FfprobePath (Join-Path $BinDir "ffprobe.exe") -Force
Write-Host "  Bundled FFmpeg:  $FfmpegPath"
Write-Host "  Bundled FFprobe: $FfprobePath"

Write-Host "[4/4] Compiling Windows installer (release\Shortly-Setup.exe)..." -ForegroundColor Cyan
$IsccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) { $Iscc = $cmd.Source }
}
if (-not $Iscc) {
    throw "Inno Setup 6 (ISCC.exe) not found. Install it with: winget install JRSoftware.InnoSetup"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "release") | Out-Null
& $Iscc (Join-Path $Root "installer\shortly.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed."
}

$OutExe = Join-Path $Root "release\Shortly-Setup.exe"
$SizeMB = [math]::Round((Get-Item $OutExe).Length / 1MB, 1)
Write-Host ""
Write-Host "SUCCESS! Installer ready: $OutExe ($SizeMB MB)" -ForegroundColor Green
