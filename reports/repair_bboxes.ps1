$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$projectRoot = Split-Path $PSScriptRoot -Parent
$outputDir = Join-Path $projectRoot 'reports/bbox_repair'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$changes = @(
    @{stem='K-001900-016548-019607-033009_0_2_0_2_70_000_200'; drug='K-016548'; old=@(88,255,366,209); bbox=@(102,862,234,236)},
    @{stem='K-003351-029667-031863_0_2_0_2_70_000_200'; drug='K-003351'; old=@(178,249,205,203); bbox=@(372,858,188,190)},
    @{stem='K-003351-020238-031863_0_2_0_2_70_000_200'; drug='K-020238'; old=@(446,844,190,191); bbox=@(584,286,220,224)},
    @{stem='K-003351-016262-018357_0_2_0_2_75_000_200'; drug='K-018357'; old=@(6567,625,311,315); bbox=@(570,622,310,318)}
)
$manifest = @()
$sheet = [System.Drawing.Bitmap]::new(976, 1360)
$graphics = [System.Drawing.Graphics]::FromImage($sheet)
$graphics.Clear([System.Drawing.Color]::White)
$pen = [System.Drawing.Pen]::new([System.Drawing.Color]::Lime, 4)
$font = [System.Drawing.Font]::new('Arial', 11)
try {
    for ($index=0; $index -lt $changes.Count; $index++) {
        $change=$changes[$index]
        $matches=@(Get-ChildItem (Join-Path $projectRoot 'data/raw/train_annotations') -Recurse -Filter ($change.stem+'.json') | Where-Object { $_.Directory.Name -eq $change.drug })
        if ($matches.Count -ne 1) { throw 'JSON target is not unique' }
        $path=$matches[0].FullName
        $raw=[IO.File]::ReadAllText($path)
        $data=$raw | ConvertFrom-Json
        if ($data.annotations.Count -ne 1 -or ($data.annotations[0].bbox -join ',') -ne ($change.old -join ',')) { throw 'Original bbox does not match; aborting' }
        $backup=Join-Path $outputDir ($change.drug+'_'+$change.stem+'.json.bak')
        if (Test-Path -LiteralPath $backup) { throw 'Backup already exists; aborting' }
        Copy-Item -LiteralPath $path -Destination $backup
        $box=$change.bbox
        $updated=[regex]::Replace($raw, '"bbox"\s*:\s*\[[^\]]*\]', ('"bbox": ['+($box -join ', ')+']'))
        $updated=[regex]::Replace($updated, '"area"\s*:\s*[0-9.]+', ('"area": '+($box[2]*$box[3])))
        [IO.File]::WriteAllText($path,$updated,[Text.UTF8Encoding]::new($false))
        $manifest += [pscustomobject]@{source=$path.Substring($projectRoot.Length+1); drug=$change.drug; old_bbox=$change.old; new_bbox=$box; method='Manual visual boundary selection; approximate pixel bounds'; backup=($backup.Substring($projectRoot.Length+1))}
        $image=[System.Drawing.Bitmap]::new((Join-Path $projectRoot ('data/raw/train_images/'+$change.stem+'.png')))
        try {
            $imageGraphics=[System.Drawing.Graphics]::FromImage($image)
            try { $imageGraphics.DrawRectangle($pen,$box[0],$box[1],$box[2],$box[3]) } finally { $imageGraphics.Dispose() }
            $column=$index%2; $row=[Math]::Floor($index/2)
            $graphics.DrawImage($image,[int]($column*488),[int]($row*680),488,640)
            $graphics.DrawString(($change.drug+' ['+($box -join ',')+']'),$font,[System.Drawing.Brushes]::Black,[single]($column*488+10),[single]($row*680+645))
        } finally { $image.Dispose() }
    }
    $sheet.Save((Join-Path $outputDir 'corrected.png'),[System.Drawing.Imaging.ImageFormat]::Png)
} finally { $graphics.Dispose(); $sheet.Dispose(); $pen.Dispose(); $font.Dispose() }
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputDir 'changes.json') -Encoding UTF8
$manifest | ConvertTo-Json -Depth 5
