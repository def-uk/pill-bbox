$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$projectRoot = Split-Path $PSScriptRoot -Parent
$outputDir = Join-Path $projectRoot 'reports/bbox_audit'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$issues = [System.Collections.Generic.List[object]]::new()
$records = [System.Collections.Generic.List[object]]::new()
$images = @{}
$dimensions = @{}
function Add-Issue([string]$kind, [string]$image, [string]$source, [string]$detail) {
    $issues.Add([pscustomobject]@{kind=$kind; image=$image; source=$source; detail=$detail})
}
$imageFiles = @(Get-ChildItem (Join-Path $projectRoot 'data/raw/train_images') -Recurse -File | Where-Object Extension -Match '^\.(png|jpe?g)$')
foreach ($file in $imageFiles) {
    if ($images.ContainsKey($file.Name)) { Add-Issue 'ambiguous_image_name' $file.Name '' $file.FullName }
    else { $images[$file.Name] = $file.FullName }
    try {
        $bitmap = [System.Drawing.Image]::FromFile($file.FullName)
        try { $dimensions[$file.Name] = @($bitmap.Width, $bitmap.Height) } finally { $bitmap.Dispose() }
    } catch { Add-Issue 'unreadable_image' $file.Name '' $_.Exception.Message }
}
$jsonFiles = @(Get-ChildItem (Join-Path $projectRoot 'data/raw/train_annotations') -Recurse -Filter '*.json' -File)
$annotationCount = 0
foreach ($file in $jsonFiles) {
    $source = $file.FullName.Substring($projectRoot.Length + 1)
    try { $data = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json }
    catch { Add-Issue 'invalid_json' '' $source $_.Exception.Message; continue }
    $localImages = @{}
    foreach ($info in $data.images) {
        if ($localImages.ContainsKey([string]$info.id)) { Add-Issue 'duplicate_image_id' $info.file_name $source ([string]$info.id) }
        $localImages[[string]$info.id] = $info
        if (!$images.ContainsKey($info.file_name)) { Add-Issue 'missing_image' $info.file_name $source '' }
        elseif ($dimensions.ContainsKey($info.file_name)) {
            $size = $dimensions[$info.file_name]
            if ($info.width -ne $size[0] -or $info.height -ne $size[1]) {
                Add-Issue 'dimension_mismatch' $info.file_name $source "JSON=$($info.width)x$($info.height); actual=$($size -join 'x')"
            }
        }
    }
    foreach ($ann in $data.annotations) {
        $annotationCount++
        if (!$localImages.ContainsKey([string]$ann.image_id)) {
            Add-Issue 'unmatched_image_id' '' $source ([string]$ann.image_id); continue
        }
        $info = $localImages[[string]$ann.image_id]
        $name = $info.file_name
        $box = @($ann.bbox)
        if ($box.Count -ne 4) { Add-Issue 'invalid_bbox' $name $source ($box -join ','); continue }
        $valid = $true
        foreach ($value in $box) {
            if ($null -eq $value -or $value -is [string] -or $value -is [bool]) { $valid = $false; break }
            try { if ([double]::IsNaN([double]$value) -or [double]::IsInfinity([double]$value)) { $valid = $false } }
            catch { $valid = $false }
        }
        if (!$valid) { Add-Issue 'invalid_bbox' $name $source ($box -join ','); continue }
        $box = @($box | ForEach-Object { [double]$_ })
        $size = @($info.width, $info.height)
        if ($dimensions.ContainsKey($name)) { $size = $dimensions[$name] }
        if ($box[2] -le 0 -or $box[3] -le 0) { Add-Issue 'nonpositive_bbox' $name $source ($box -join ',') }
        if ($box[0] -lt 0 -or $box[1] -lt 0 -or ($box[0]+$box[2]) -gt $size[0] -or ($box[1]+$box[3]) -gt $size[1]) {
            Add-Issue 'out_of_bounds' $name $source ($box -join ',')
        }
        $records.Add([pscustomobject]@{image=$name; source=$source; drug=$info.drug_N; name=$info.dl_name; bbox=$box})
    }
}
$groups = @{}
foreach ($record in $records) {
    if (!$groups.ContainsKey($record.image)) { $groups[$record.image] = [System.Collections.Generic.List[object]]::new() }
    $groups[$record.image].Add($record)
}
foreach ($name in $groups.Keys) {
    $items = $groups[$name]
    for ($i=0; $i -lt $items.Count; $i++) {
        for ($j=$i+1; $j -lt $items.Count; $j++) {
            $a=$items[$i].bbox; $b=$items[$j].bbox
            $detail = "$($items[$i].drug) [$($a -join ',')] / $($items[$j].drug) [$($b -join ',')]; other=$($items[$j].source)"
            if (($a -join ',') -eq ($b -join ',')) { Add-Issue 'duplicate_bbox' $name $items[$i].source $detail; continue }
            $width=[Math]::Max(0, [Math]::Min($a[0]+$a[2],$b[0]+$b[2])-[Math]::Max($a[0],$b[0]))
            $height=[Math]::Max(0, [Math]::Min($a[1]+$a[3],$b[1]+$b[3])-[Math]::Max($a[1],$b[1]))
            $intersection=$width*$height
            $union=$a[2]*$a[3]+$b[2]*$b[3]-$intersection
            if ($union -gt 0 -and $intersection/$union -ge 0.9) { Add-Issue 'high_overlap_review' $name $items[$i].source $detail }
        }
    }
}
foreach ($name in $images.Keys) { if (!$groups.ContainsKey($name)) { Add-Issue 'no_valid_annotation' $name '' '' } }
$issues | Export-Csv (Join-Path $outputDir 'issues.csv') -NoTypeInformation -Encoding UTF8
$summary = [ordered]@{image_files=$imageFiles.Count; json_files=$jsonFiles.Count; annotations=$annotationCount; issue_rows=$issues.Count; affected_images=@($issues | Where-Object image | Select-Object -ExpandProperty image -Unique).Count; by_kind=@($issues | Group-Object kind | Select-Object Name,Count)}
$summary | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputDir 'summary.json') -Encoding UTF8
$html = [System.Text.StringBuilder]::new()
[void]$html.Append('<!doctype html><meta charset="utf-8"><title>Bbox audit</title><style>body{font-family:sans-serif;margin:24px}article{margin-bottom:40px}svg{width:480px;max-width:100%;background:#eee}text{paint-order:stroke;stroke:white;stroke-width:3px;fill:black;font-size:20px}pre{white-space:pre-wrap}</style><h1>Bbox audit</h1><p>Duplicate and high-overlap boxes require visual review. All boxes are numbered. Coordinates are not automatically corrected.</p>')
foreach ($group in ($issues | Where-Object image | Group-Object image)) {
    $name=$group.Name
    [void]$html.Append('<article><h3>'+[System.Net.WebUtility]::HtmlEncode($name)+'</h3><pre>'+[System.Net.WebUtility]::HtmlEncode(($group.Group | ForEach-Object { "$($_.kind): $($_.detail)" } | Out-String))+'</pre>')
    if ($images.ContainsKey($name) -and $dimensions.ContainsKey($name)) {
        $size=$dimensions[$name]
        $relative='../../'+$images[$name].Substring($projectRoot.Length+1).Replace('\','/')
        [void]$html.Append("<svg viewBox='0 0 $($size[0]) $($size[1])'><image href='$relative' width='$($size[0])' height='$($size[1])'/>")
        $index=0
        foreach ($record in $groups[$name]) {
            if ($null -eq $record) { continue }
            $index++; $box=$record.bbox
            $color=@('red','lime','cyan','orange','magenta')[($index-1)%5]
            [void]$html.Append("<rect x='$($box[0])' y='$($box[1])' width='$($box[2])' height='$($box[3])' fill='none' stroke='$color' stroke-width='4'/><text x='$($box[0])' y='$($box[1]+22*$index)'>$index $($record.drug)</text>")
        }
        [void]$html.Append('</svg>')
    }
    [void]$html.Append('</article>')
}
$html.ToString() | Set-Content (Join-Path $outputDir 'review.html') -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
