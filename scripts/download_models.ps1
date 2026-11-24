# Resumable download of HF model repos at a pinned commit into G:\models\<org>\<name>.
# Each file goes to .part, is checked against HF size (and sha256 for LFS files), then renamed.
param([string]$Repos)  # comma-separated org/name@commit (powershell -File passes arrays as one string)
$ErrorActionPreference = 'Continue'
$specs = @($Repos -split '[,\s]+' | Where-Object { $_ })
$want = '\.(json|safetensors|txt|model|jinja)$|^LICENSE|^README'
foreach ($spec in $specs) {
  $repo, $rev = $spec -split '@'
  $dest = Join-Path 'G:\models' ($repo -replace '/', '\')
  New-Item -ItemType Directory -Force $dest | Out-Null
  $tree = Invoke-RestMethod "https://huggingface.co/api/models/$repo/tree/${rev}?recursive=true"
  $files = @($tree | Where-Object { $_.type -eq 'file' -and ($_.path -split '/')[-1] -match $want })
  $total = ($files | Measure-Object size -Sum).Sum
  "REPO $repo@$rev  $($files.Count) files  {0:N2} GB  $(Get-Date -Format s)" -f ($total / 1GB)
  foreach ($f in $files) {
    $final = Join-Path $dest ($f.path -replace '/', '\')
    New-Item -ItemType Directory -Force (Split-Path $final) | Out-Null
    if ((Test-Path $final) -and (Get-Item $final).Length -eq $f.size) { "  SKIP $($f.path)"; continue }
    $part = "$final.part"
    $url = "https://huggingface.co/$repo/resolve/$rev/$($f.path)"
    for ($try = 1; $try -le 5; $try++) {
      & curl.exe -L -sS -f -C - --retry 5 --retry-delay 10 -o $part $url
      if ((Test-Path $part) -and (Get-Item $part).Length -ge $f.size) { break }
      "  retry $try for $($f.path) (curl exit $LASTEXITCODE)"; Start-Sleep 20
    }
    $len = if (Test-Path $part) { (Get-Item $part).Length } else { 0 }
    if ($len -ne $f.size) { "  FAIL $($f.path) size $len expected $($f.size)"; continue }
    if ($f.lfs) {
      $sha = (Get-FileHash $part -Algorithm SHA256).Hash.ToLower()
      if ($sha -ne $f.lfs.oid) { "  FAIL $($f.path) sha256 mismatch; left as .part"; continue }
    }
    Move-Item -Force $part $final
    "  OK $($f.path) {0:N2} GB $(Get-Date -Format s)" -f ($f.size / 1GB)
  }
  "DONE $repo $(Get-Date -Format s)"
}
"ALL DONE $(Get-Date -Format s)"
