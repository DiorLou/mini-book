param(
    [string]$PublishDir = "",
    [switch]$ChangedOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LocalMyst = Join-Path $RepoRoot ".venv/Scripts/myst.exe"
$MystCommand = if (Test-Path -LiteralPath $LocalMyst) {
    $LocalMyst
}
else {
    (Get-Command myst -ErrorAction Stop).Source
}
$Projects = @(
    @{ Path = "deep learning"; Slug = "computer"; Pdf = "computer-notes.pdf" },
    @{ Path = "finance"; Slug = "finance"; Pdf = "finance-notes.pdf" },
    @{ Path = "personal"; Slug = "resume"; Pdf = "personal-resume.pdf" }
)

if ($ChangedOnly -and -not $PublishDir) {
    throw "-ChangedOnly requires -PublishDir so previous build state can be tracked."
}

function Get-ProjectFingerprint {
    param([string]$ProjectPath)

    $InputFiles = @(
        Get-ChildItem -LiteralPath $ProjectPath -File -Recurse |
            Where-Object { $_.FullName -notlike "*$([IO.Path]::DirectorySeparatorChar)_build$([IO.Path]::DirectorySeparatorChar)*" }
        Get-ChildItem -LiteralPath (Join-Path $RepoRoot "assets/fonts") -File -Recurse -ErrorAction SilentlyContinue
        Get-Item -LiteralPath $PSCommandPath
    ) | Sort-Object FullName

    $FingerprintSource = foreach ($File in $InputFiles) {
        $RelativePath = $File.FullName.Substring($RepoRoot.Length).TrimStart("\", "/").Replace("\", "/")
        "$RelativePath`n$((Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash)"
    }

    $Bytes = [Text.Encoding]::UTF8.GetBytes(($FingerprintSource -join "`n"))
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($Hasher.ComputeHash($Bytes))).Replace("-", "")
    }
    finally {
        $Hasher.Dispose()
    }
}

$BuildStatePath = if ($PublishDir) { Join-Path $PublishDir ".mini-book-build-state.json" } else { $null }
$PreviousBuildState = @{}
if ($ChangedOnly -and (Test-Path -LiteralPath $BuildStatePath)) {
    $SavedState = Get-Content -LiteralPath $BuildStatePath -Raw | ConvertFrom-Json
    foreach ($Property in $SavedState.PSObject.Properties) {
        $PreviousBuildState[$Property.Name] = $Property.Value
    }
}

$CurrentBuildState = @{}

$PreviousNodeOptions = [Environment]::GetEnvironmentVariable("NODE_OPTIONS", "Process")
$PreviousTypstFontPaths = [Environment]::GetEnvironmentVariable("TYPST_FONT_PATHS", "Process")
$DeprecationFilter = "--disable-warning=DEP0169"
if ($PreviousNodeOptions -notlike "*$DeprecationFilter*") {
    $env:NODE_OPTIONS = (($PreviousNodeOptions, $DeprecationFilter) -join " ").Trim()
}

# Use repository-owned static fonts so PDF output is reproducible on Windows,
# other development machines, and GitHub Actions runners.
$env:TYPST_FONT_PATHS = Join-Path $RepoRoot "assets/fonts"

try {
    foreach ($Project in $Projects) {
        $ProjectPath = Join-Path $RepoRoot $Project.Path
        $Fingerprint = Get-ProjectFingerprint -ProjectPath $ProjectPath
        $CurrentBuildState[$Project.Slug] = $Fingerprint
        $Destination = if ($PublishDir) { Join-Path $PublishDir $Project.Slug } else { $null }
        $PublishedIndex = if ($Destination) { Join-Path $Destination "index.html" } else { $null }
        $ShouldBuild = -not $ChangedOnly -or
            $PreviousBuildState[$Project.Slug] -ne $Fingerprint -or
            -not (Test-Path -LiteralPath $PublishedIndex)

        if (-not $ShouldBuild) {
            Write-Host "Skipping $($Project.Slug) (unchanged)."
            continue
        }

        Write-Host "Building $($Project.Slug)..."

        # GitHub Pages hosts this repository below /<repository>/<book>/.
        # MyST uses BASE_URL to generate links and asset paths for that location.
        if ($env:GITHUB_REPOSITORY) {
            $RepositoryName = ($env:GITHUB_REPOSITORY -split "/")[-1]
            $env:BASE_URL = "/$RepositoryName/$($Project.Slug)"
        }
        elseif ($PublishDir) {
            $env:BASE_URL = "/$($Project.Slug)"
        }
        else {
            Remove-Item Env:BASE_URL -ErrorAction SilentlyContinue
        }

        Push-Location $ProjectPath
        try {
            & $MystCommand build --typst --ci
            $PdfPath = Join-Path $ProjectPath "_build/exports/$($Project.Pdf)"
            if (-not (Test-Path $PdfPath)) {
                throw "PDF was not generated: $PdfPath. Review the MyST/Typst error shown above."
            }
            & $MystCommand build --html --ci
        }
        finally {
            Pop-Location
        }

        if ($PublishDir) {
            New-Item -ItemType Directory -Force -Path $Destination | Out-Null
            Copy-Item -Path (Join-Path $ProjectPath "_build/html/*") -Destination $Destination -Recurse -Force
        }
    }

    if ($PublishDir) {
        New-Item -ItemType Directory -Force -Path $PublishDir | Out-Null
        Copy-Item -Path (Join-Path $RepoRoot "portal/*") -Destination $PublishDir -Recurse -Force
        $CurrentBuildState | ConvertTo-Json | Set-Content -LiteralPath $BuildStatePath -Encoding utf8
        Write-Host "Combined website created at $PublishDir"
    }
}
finally {
    [Environment]::SetEnvironmentVariable("NODE_OPTIONS", $PreviousNodeOptions, "Process")
    [Environment]::SetEnvironmentVariable("TYPST_FONT_PATHS", $PreviousTypstFontPaths, "Process")
}
