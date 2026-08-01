param(
    [string]$PublishDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Projects = @(
    @{ Path = "deep learning"; Slug = "computer"; Pdf = "computer-notes.pdf" },
    @{ Path = "finance"; Slug = "finance"; Pdf = "finance-notes.pdf" },
    @{ Path = "personal"; Slug = "resume"; Pdf = "personal-resume.pdf" }
)

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
            myst build --typst --ci
            $PdfPath = Join-Path $ProjectPath "_build/exports/$($Project.Pdf)"
            if (-not (Test-Path $PdfPath)) {
                throw "PDF was not generated: $PdfPath. Review the MyST/Typst error shown above."
            }
            myst build --html --ci
        }
        finally {
            Pop-Location
        }

        if ($PublishDir) {
            $Destination = Join-Path $PublishDir $Project.Slug
            New-Item -ItemType Directory -Force -Path $Destination | Out-Null
            Copy-Item -Path (Join-Path $ProjectPath "_build/html/*") -Destination $Destination -Recurse -Force
        }
    }

    if ($PublishDir) {
        Copy-Item -Path (Join-Path $RepoRoot "portal/*") -Destination $PublishDir -Recurse -Force
        Write-Host "Combined website created at $PublishDir"
    }
}
finally {
    [Environment]::SetEnvironmentVariable("NODE_OPTIONS", $PreviousNodeOptions, "Process")
    [Environment]::SetEnvironmentVariable("TYPST_FONT_PATHS", $PreviousTypstFontPaths, "Process")
}
