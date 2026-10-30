$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$models = @(
    @{ id='distilbert/distilbert-base-uncased'; folder='distilbert'; revision='12040accade4e8a0f71eabdb258fecc2e7e948be'; files=@('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','LICENSE','README.md') },
    @{ id='microsoft/trocr-small-handwritten'; folder='trocr'; revision='b4648cfa171985a6745f37ddd637e98c0da958ac'; files=@('config.json','generation_config.json','preprocessor_config.json','pytorch_model.bin','sentencepiece.bpe.model','special_tokens_map.json','tokenizer_config.json','README.md') }
)
$manifest = @()
foreach ($model in $models) {
    $dest = Join-Path $root "data/pretrained/$($model.folder)"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    foreach ($file in $model.files) {
        $target = Join-Path $dest $file
        if (-not (Test-Path -LiteralPath $target)) {
            Write-Output "Downloading $($model.id)/$file"
            $done = $false
            for ($attempt=1; $attempt -le 3; $attempt++) {
                try {
                    Invoke-WebRequest -Uri "https://huggingface.co/$($model.id)/resolve/$($model.revision)/${file}?download=true" -OutFile "$target.partial" -TimeoutSec 600
                    $done = $true
                    break
                } catch { Write-Output "Retry $attempt for $file : $($_.Exception.Message)" }
            }
            if (-not $done) { throw "Could not download $file" }
            Move-Item -LiteralPath "$target.partial" -Destination $target
        }
        $manifest += @{ model=$model.id; revision=$model.revision; file=$file; sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash; bytes=(Get-Item -LiteralPath $target).Length }
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'data/pretrained/manifest.json') -Encoding utf8
