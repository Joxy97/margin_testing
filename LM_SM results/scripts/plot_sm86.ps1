param(
    [int]$VariableCount = 200,
    [ValidateSet('sparse', 'dense')][string]$DensityCategory = 'dense',
    [string]$BaseDir = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'PLots')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$allowedCounts = @(100, 200, 500, 1000, 5000, 10000)
if ($allowedCounts -notcontains $VariableCount) { throw "Unsupported variable count: $VariableCount" }
if ($DensityCategory -eq 'dense' -and $VariableCount -notin @(100, 200, 500, 1000, 5000)) {
    throw "The dense result for $VariableCount variables is not present locally."
}
$runNames = @("${DensityCategory}_$VariableCount", "codev4_${VariableCount}_$DensityCategory")
$resultRoots = @($runNames | ForEach-Object { Join-Path $BaseDir "$_\SM86" } |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ 'manifest.json') })
if ($resultRoots.Count -ne 1) { throw "Expected exactly one SM86 result folder in $BaseDir, found $($resultRoots.Count)." }
$ResultRoot = $resultRoots[0]
$manifest = Get-Content -Raw -LiteralPath (Join-Path $ResultRoot 'manifest.json') | ConvertFrom-Json
if ($manifest.schema_version -ne 'sm86_collection_v1' -or $manifest.resumable -ne $false) {
    throw 'Expected the explicit derived, non-resumable SM86 collection manifest.'
}
$problem = @($manifest.problems)[0]
if (@($manifest.problems).Count -ne 1 -or [int]$problem.n -ne $VariableCount -or $problem.density -ne $DensityCategory) {
    throw "Expected one completed $VariableCount-variable $DensityCategory result."
}
$catalogPath = Join-Path $ResultRoot 'reproducibility\campaign\catalog.json'
$catalogHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $catalogPath).Hash
$catalogHashes = @($manifest.partitions | ForEach-Object { [string]$_.catalog } | Sort-Object -Unique)
if (@($manifest.partitions).Count -ne 4 -or $catalogHashes.Count -ne 1 -or $catalogHash -ne $catalogHashes[0]) {
    throw "Catalog snapshot does not match all four original manifests: $catalogPath"
}
$catalog = Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json
$catalogEntries = @($catalog.instances | Where-Object {
    $_.instance_id -eq $problem.id -and [int]$_.binary_variables -eq $VariableCount -and
    $_.density_category -eq $DensityCategory
})
if ($catalogEntries.Count -ne 1) { throw "Missing or ambiguous density entry for $($problem.id)." }
$entry = $catalogEntries[0]
$measuredDensity = 100.0 * [double]$entry.nonzero_offdiagonal_pairs / ($VariableCount * ($VariableCount - 1) / 2.0)
if ([Math]::Abs($measuredDensity - [double]$entry.actual_density_percent) -gt 1e-12) {
    throw "Catalog density and pair count disagree for $($problem.id)."
}
$densityLabel = $measuredDensity.ToString('0.00', $culture) + "% QUBO pair density ($DensityCategory)"
$referenceType = [string]$problem.reference_type
if ($referenceType -notin @('OPTIMUM', 'BKS')) { throw "Unexpected reference type: $referenceType" }
$referenceName = if ($referenceType -eq 'OPTIMUM') { 'optimum' } else { 'best-known value' }
$hitName = if ($referenceType -eq 'OPTIMUM') { 'Exact optimum' } else { 'BKS or better' }
$qualityReferenceName = if ($referenceType -eq 'OPTIMUM') { 'published optimum' } else { 'published best-known solution' }
$reference = [double]$problem.reference
$nearTarget = $reference + 0.01 * [Math]::Abs($reference)
$exactOnlyWindow = $referenceType -eq 'OPTIMUM' -and [bool]$problem.integer_objective -and
    ($nearTarget - $reference) -lt 1.0
$withinUnit = if ($exactOnlyWindow) {
    'Only exact optimum qualifies: integer objective, 1% < one energy unit'
} else {
    'Qualifying seeds out of 100; reference hits included'
}
$checkpoints = @($manifest.checkpoints_seconds | ForEach-Object { [double]$_ })
$seedCount = @($manifest.seeds).Count
$solverNames = @(Get-ChildItem -LiteralPath (Join-Path $ResultRoot 'aggregated') -Directory | Select-Object -ExpandProperty Name | Sort-Object)
$expectedSolvers = @(
    'lib_c3_dzyaloshinskii_moriya',
    'lib_c3_resonant_dipole_exchange',
    'lib_c3_lipkin_meshkov_glick',
    'lib_c3_rydberg_pxp',
    'lib_c3_aubry_andre',
    'lib_c3_haldane',
    'lib_c3_long_range_transverse_ising',
    'lib_c3_frustrated_j1_j2',
    'lib_c3_bilinear_biquadratic',
    'lib_c3_kitaev_heisenberg_gamma',
    'lib_c3_harper_hofstadter',
    'lib_c3_holstein',
    'lib_c3_rydberg_dressed_ising',
    'lib_c3_bose_fermi_mixture',
    'lib_c3_degenerate_parametric_amplifier',
    'lib_c3_trapped_ion_spin_phonon',
    'lib_c3_molmer_sorensen',
    'lib_c3_kanamori',
    'lib_c3_anderson_impurity',
    'lib_c3_pariser_parr_pople',
    'lib_c3_scalar_spin_chirality',
    'lib_c3_ring_exchange',
    'lib_c3_central_spin',
    'lib_c3_frohlich_polaron',
    'lib_c3_lieb_liniger',
    'lib_c3_bosonic_josephson',
    'lib_c3_spin_dependent_optical_lattice',
    'lib_c3_optical_clock',
    'lib_c3_dipolar_xxz',
    'lib_c3_kugel_khomskii',
    'lib_c3_kondo_lattice',
    'lib_c3_hubbard_holstein',
    'lib_c3_pair_hopping_bose_hubbard',
    'lib_c3_quantum_east',
    'lib_c3_quantum_p_spin',
    'lib_c3_kicked_ising',
    'lib_c3_wen_plaquette',
    'lib_c3_schwinger',
    'lib_c3_plaquette_ising',
    'lib_c3_spin_transfer_torque',
    'lib_c3_levin_wen',
    'lib_c3_double_semion',
    'lib_c3_quantum_link',
    'lib_c3_majumdar_ghosh',
    'lib_c3_creutz_ladder',
    'lib_c3_penson_kolb',
    'lib_c3_richardson_gaudin',
    'lib_c3_spinor_rydberg',
    'lib_c3_quantum_hopfield',
    'lib_c3_quantum_potts',
    'lib_c3_frustrated_spin_glass',
    'lib_c3_blume_capel',
    'lib_c3_ashkin_teller',
    'lib_c3_blume_emery_griffiths',
    'lib_c3_fano_anderson',
    'lib_c3_feshbach_two_channel',
    'lib_c3_optical_lattice',
    'lib_c3_dirac_coulomb_breit',
    'lib_c3_xyz_spin',
    'lib_c3_xx_spin',
    'lib_c3_xxz_spin',
    'lib_c3_kitaev_honeycomb',
    'lib_c3_quantum_compass',
    'lib_c3_rydberg_atom',
    'lib_c3_rice_mele',
    'lib_c3_rashba_bdg',
    'lib_c3_spin_orbit_bose_gas',
    'lib_c3_raman_spin_orbit',
    'lib_c3_tavis_cummings',
    'lib_c3_lambda_three_level',
    'lib_c3_quantum_rabi',
    'lib_c3_weyl_spin_orbit',
    'lib_c3_rashba_zeeman',
    'lib_c3_persistent_spin_helix',
    'lib_c3_rashba_dresselhaus',
    'lib_c3_cubic_dresselhaus',
    'lib_c3_dresselhaus',
    'lib_c3_rashba'
) | Sort-Object
if ($solverNames.Count -ne 78 -or (Compare-Object $expectedSolvers $solverNames)) {
    throw 'Expected exactly the 78 compatible Compendium III solvers in aggregated results.'
}
if (Compare-Object $expectedSolvers @($manifest.solvers.PSObject.Properties.Name | Sort-Object)) {
    throw 'Manifest solver inventory disagrees with the 78 compatible Compendium III solvers.'
}
if ($seedCount -ne 100 -or (Compare-Object @(0..99) @($manifest.seeds))) {
    throw 'Expected independent seeds 0 through 99 for every solver.'
}
$expectedCheckpoints = @(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
if ($checkpoints.Count -ne 9) { throw 'Expected the existing nine checkpoint times.' }
for ($i = 0; $i -lt 9; $i++) {
    if ($checkpoints[$i] -ne $expectedCheckpoints[$i]) { throw 'Unexpected checkpoint schedule.' }
}
$withinCsv = Join-Path $ResultRoot 'postprocessed\hits_within_1pct.csv'
$withinRows = if (Test-Path -LiteralPath $withinCsv) { @(Import-Csv -LiteralPath $withinCsv) } else { @() }
$withinLookup = @{}
foreach ($row in $withinRows) {
    $withinLookup["$($row.solver)|$($row.checkpoint_seconds)"] = [int]$row.hits
}

function Read-NpyFloat64([string]$path, [int]$expectedRows, [int]$expectedColumns) {
    $bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $path))
    if ($bytes.Length -lt 12 -or $bytes[0] -ne 147 -or
        [Text.Encoding]::ASCII.GetString($bytes, 1, 5) -ne 'NUMPY') {
        throw "Invalid NPY file: $path"
    }
    if ($bytes[6] -eq 1) {
        $headerLength = [int][BitConverter]::ToUInt16($bytes, 8)
        $start = 10
    } elseif ($bytes[6] -eq 2 -or $bytes[6] -eq 3) {
        $headerLength = [int][BitConverter]::ToUInt32($bytes, 8)
        $start = 12
    } else {
        throw "Unsupported NPY version: $path"
    }
    $header = [Text.Encoding]::ASCII.GetString($bytes, $start, $headerLength)
    $shape = [regex]::Match($header, "'shape':\s*\((\d+),\s*(\d+)\)")
    if ($header -notmatch "'descr':\s*'<f8'" -or
        $header -notmatch "'fortran_order':\s*False" -or
        -not $shape.Success -or
        [int]$shape.Groups[1].Value -ne $expectedRows -or
        [int]$shape.Groups[2].Value -ne $expectedColumns) {
        throw "Unexpected NPY dtype, layout, or shape: $path"
    }
    $dataStart = $start + $headerLength
    if ($bytes.Length -ne $dataStart + 8 * $expectedRows * $expectedColumns) {
        throw "Wrong NPY payload size: $path"
    }
    return [pscustomobject]@{ Bytes = $bytes; DataStart = $dataStart }
}

$series = @{}
foreach ($solver in $solverNames) {
    $csv = Join-Path $ResultRoot "aggregated\$solver\$($problem.id).csv"
    $rawPath = Join-Path $ResultRoot "raw\$solver\$($problem.id).npy"
    $raw = Read-NpyFloat64 $rawPath $seedCount $checkpoints.Count
    $exactCounts = New-Object 'int[]' $checkpoints.Count
    $withinCounts = New-Object 'int[]' $checkpoints.Count
    $availableCounts = New-Object 'int[]' $checkpoints.Count
    $sums = New-Object 'double[]' $checkpoints.Count
    for ($seed = 0; $seed -lt $seedCount; $seed++) {
        for ($index = 0; $index -lt $checkpoints.Count; $index++) {
            $position = $raw.DataStart + 8 * ($seed * $checkpoints.Count + $index)
            $energy = [BitConverter]::ToDouble($raw.Bytes, $position)
            if ([double]::IsNaN($energy)) { continue }
            if ([double]::IsInfinity($energy)) { throw "Nonfinite energy in $rawPath at seed $seed, checkpoint $index" }
            if ($referenceType -eq 'OPTIMUM' -and $energy -lt $reference) {
                throw "Energy below the proven optimum in $rawPath at seed $seed, checkpoint $index`: $energy."
            }
            $availableCounts[$index]++
            $sums[$index] += $energy
            if ($energy -le $reference) { $exactCounts[$index]++ }
            if ($energy -le $nearTarget) { $withinCounts[$index]++ }
        }
    }
    $rows = @(Import-Csv -LiteralPath $csv)
    if ($exactOnlyWindow) {
        for ($index = 0; $index -lt $checkpoints.Count; $index++) {
            if ($withinCounts[$index] -ne $exactCounts[$index]) {
                throw "Noninteger energy entered the exact-only 1% window in $rawPath at checkpoint $index"
            }
        }
    }
    if ($rows.Count -ne $checkpoints.Count) { throw "Wrong checkpoint count: $csv" }
    $points = @()
    for ($index = 0; $index -lt $rows.Count; $index++) {
        $row = $rows[$index]
        $t = [double]::Parse($row.checkpoint_seconds, $culture)
        if ($t -ne $checkpoints[$index]) { throw "Checkpoint order disagrees with manifest: $csv" }
        $p = [double]::Parse($row.hit_probability, $culture)
        $gap = [double]::Parse($row.gap_to_solution, $culture)
        $key = "$solver|$($row.checkpoint_seconds)"
        if ($withinRows.Count -gt 0 -and -not $withinLookup.ContainsKey($key)) { throw "Missing 1% hits: $key" }
        if (($withinRows.Count -gt 0 -and $withinLookup[$key] -ne $withinCounts[$index]) -or
            [int][Math]::Round($seedCount * $p) -ne $exactCounts[$index]) {
            throw "Stored hit counts disagree with raw energies: $key"
        }
        $expectedGap = if ($availableCounts[$index] -eq $seedCount) {
            $sums[$index] / $seedCount - $reference
        } else { [double]::NaN }
        if (([double]::IsNaN($expectedGap) -ne [double]::IsNaN($gap)) -or
            (-not [double]::IsNaN($expectedGap) -and [Math]::Abs($expectedGap - $gap) -gt 1e-7)) {
            throw "Stored mean gap disagrees with raw energies: $key"
        }
        $gapPercent = 100 * $gap / [Math]::Abs($reference)
        if ($referenceType -eq 'OPTIMUM' -and -not [double]::IsNaN($gapPercent) -and $gapPercent -lt -1e-9) {
            throw "Negative reference gap in $csv at checkpoint $t"
        }
        $points += [pscustomobject]@{
            Time = $t
            ExactHits = $exactCounts[$index]
            WithinHits = $withinCounts[$index]
            GapPercent = $gapPercent
        }
    }
    $series[$solver] = $points
}

$colors = @(
    [System.Drawing.Color]::FromArgb(0,114,178),
    [System.Drawing.Color]::FromArgb(213,94,0),
    [System.Drawing.Color]::FromArgb(0,158,115),
    [System.Drawing.Color]::FromArgb(204,121,167),
    [System.Drawing.Color]::FromArgb(230,159,0),
    [System.Drawing.Color]::FromArgb(86,180,233),
    [System.Drawing.Color]::FromArgb(0,0,0)
)
$ink = [System.Drawing.Color]::FromArgb(37,47,61)
$muted = [System.Drawing.Color]::FromArgb(95,105,117)
$grid = [System.Drawing.Color]::FromArgb(224,229,235)
$panelBorder = [System.Drawing.Color]::FromArgb(209,217,225)
$titleFont = New-Object System.Drawing.Font('Segoe UI', 21, [System.Drawing.FontStyle]::Bold)
$panelFont = New-Object System.Drawing.Font('Segoe UI', 13, [System.Drawing.FontStyle]::Bold)
$axisFont = New-Object System.Drawing.Font('Segoe UI', 9)
$legendFont = New-Object System.Drawing.Font('Segoe UI', 10)
$subtitleFont = New-Object System.Drawing.Font('Segoe UI', 11)
$formatCenter = New-Object System.Drawing.StringFormat
$formatCenter.Alignment = [System.Drawing.StringAlignment]::Center
$formatRight = New-Object System.Drawing.StringFormat
$formatRight.Alignment = [System.Drawing.StringAlignment]::Far

# Preserve the existing SM20/SM23 typography, plot area, lines and grid. The
# canvas is 160px taller solely to fit seven legend rows without shrinking them.
function Draw-Figure([string]$metric, [string]$title, [string]$unit, [string]$fileName,
                     [int]$part, [int]$pageStart, [int]$solversPerPanel) {
    $bitmap = New-Object System.Drawing.Bitmap 2250, 1640
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)
    $inkBrush = New-Object System.Drawing.SolidBrush($ink)
    $mutedBrush = New-Object System.Drawing.SolidBrush($muted)
    $gridPen = New-Object System.Drawing.Pen($grid, 1)
    $borderPen = New-Object System.Drawing.Pen($panelBorder, 1)
    try {
        $graphics.DrawString("$title | SM86, part $part of 2", $titleFont, $inkBrush, [float]54, [float]26)
        $graphics.DrawString("$($problem.id) | $VariableCount variables | $densityLabel | published $referenceName $([int]$reference) | 100 seeds per solver | 78 compatible models; 8 unsupported", $subtitleFont, $mutedBrush, [float]55, [float]67)
        $graphics.DrawString('Checkpoint time (seconds; logarithmic x-axis)', $subtitleFont, $mutedBrush, [float]55, [float]1583)
        $graphics.DrawString('QUBO pair density = nonzero unordered off-diagonal pairs / all possible pairs', $axisFont, $mutedBrush, [float]55, [float]1610)
        for ($panel = 0; $panel -lt 6; $panel++) {
            $column = $panel % 3
            $rowNumber = [Math]::Floor($panel / 3)
            $px = [int](45 + 735 * $column)
            $py = [int](112 + 725 * $rowNumber)
            $panelRect = New-Object System.Drawing.Rectangle($px, $py, 705, 700)
            $graphics.DrawRectangle($borderPen, $panelRect)
            $start = $pageStart + $panel * $solversPerPanel
            $end = [Math]::Min($start + $solversPerPanel - 1, $solverNames.Count - 1)
            if ($start -ge $solverNames.Count) { throw 'Unexpected empty SM86 figure panel.' }
            $graphics.DrawString("Solvers $($start + 1)-$($end + 1)", $panelFont, $inkBrush, [float]($px + 23), [float]($py + 15))
            $graphics.DrawString($unit, $axisFont, $mutedBrush, [float]($px + 23), [float]($py + 47))
            $plotX = $px + 70
            $plotY = $py + 78
            $plotW = 605
            $plotH = 390
            $yMin = 0.0
            $yMax = 100.0
            if ($metric -eq 'GapPercent') {
                $finite = @()
                for ($idx = $start; $idx -le $end; $idx++) {
                    foreach ($point in $series[$solverNames[$idx]]) {
                        $value = [double]$point.$metric
                        if (-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value)) {
                            $finite += $value
                        }
                    }
                }
                $bounds = $finite | Measure-Object -Minimum -Maximum
                if ($bounds.Minimum -lt 0) { $yMin = [Math]::Floor($bounds.Minimum * 1.08 * 2) / 2 }
                $peak = $bounds.Maximum
                $yMax = [Math]::Max(2.0, [Math]::Ceiling($peak * 1.08 * 2) / 2)
            }
            for ($tick = 0; $tick -le 5; $tick++) {
                $fraction = $tick / 5.0
                $y = [float]($plotY + $plotH * (1 - $fraction))
                $graphics.DrawLine($gridPen, [float]$plotX, $y, [float]($plotX + $plotW), $y)
                $label = if ($metric -eq 'GapPercent') {
                    $tickValue = $yMin + ($yMax - $yMin) * $fraction
                    if ([Math]::Abs($tickValue) -ge 10000) { '{0:0.##E+0}%' -f $tickValue }
                    else { '{0:0.##}%' -f $tickValue }
                } else { [string][int](100 * $fraction) }
                $graphics.DrawString($label, $axisFont, $mutedBrush,
                    (New-Object System.Drawing.RectangleF([float]($plotX - 71), [float]($y - 8), [float]64, [float]18)), $formatRight)
            }
            if ($metric -eq 'GapPercent') {
                $guideY = [float]($plotY + $plotH * (1 - (1 - $yMin) / ($yMax - $yMin)))
                $guidePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(84, 91, 101), 1.8)
                $guidePen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
                try { $graphics.DrawLine($guidePen, [float]$plotX, $guideY, [float]($plotX + $plotW), $guideY) }
                finally { $guidePen.Dispose() }
            }
            foreach ($time in $checkpoints) {
                $xFraction = ([Math]::Log10($time) - [Math]::Log10($checkpoints[0])) /
                    ([Math]::Log10($checkpoints[-1]) - [Math]::Log10($checkpoints[0]))
                $x = [float]($plotX + $plotW * $xFraction)
                $graphics.DrawLine($gridPen, $x, [float]$plotY, $x, [float]($plotY + $plotH))
                $graphics.DrawString(($time.ToString('0.##', $culture)), $axisFont, $mutedBrush,
                    (New-Object System.Drawing.RectangleF([float]($x - 25), [float]($plotY + $plotH + 8), [float]50, [float]20)), $formatCenter)
            }
            for ($idx = $start; $idx -le $end; $idx++) {
                $solver = $solverNames[$idx]
                $color = $colors[$idx - $start]
                $pen = New-Object System.Drawing.Pen($color, 2.8)
                $brush = New-Object System.Drawing.SolidBrush($color)
                try {
                    $previous = $null
                    foreach ($point in $series[$solver]) {
                        $value = [double]$point.$metric
                        if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { $previous = $null; continue }
                        $xFraction = ([Math]::Log10($point.Time) - [Math]::Log10($checkpoints[0])) /
                            ([Math]::Log10($checkpoints[-1]) - [Math]::Log10($checkpoints[0]))
                        $x = [float]($plotX + $plotW * $xFraction)
                        $y = [float]($plotY + $plotH * (1 - ($value - $yMin) / ($yMax - $yMin)))
                        if ($null -ne $previous) { $graphics.DrawLine($pen, $previous.X, $previous.Y, $x, $y) }
                        $graphics.FillEllipse($brush, [float]($x - 3.5), [float]($y - 3.5), [float]7, [float]7)
                        $previous = [pscustomobject]@{ X = $x; Y = $y }
                    }
                    $legendY = [float]($py + 506 + 26 * ($idx - $start))
                    $graphics.DrawLine($pen, [float]($px + 28), $legendY, [float]($px + 56), $legendY)
                    $graphics.FillEllipse($brush, [float]($px + 39), [float]($legendY - 3.5), [float]7, [float]7)
                    $graphics.DrawString($solver.Replace('lib_', ''), $legendFont, $inkBrush, [float]($px + 66), [float]($legendY - 10))
                } finally { $pen.Dispose(); $brush.Dispose() }
            }
        }
        $out = Join-Path $OutputDir $fileName
        $bitmap.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
        Write-Output $out
        $internal = Join-Path $ResultRoot "plots\$($problem.id)"
        New-Item -ItemType Directory -Path $internal -Force | Out-Null
        $copy = Join-Path $internal $fileName
        if ([IO.Path]::GetFullPath($copy) -ne [IO.Path]::GetFullPath($out)) {
            Copy-Item -LiteralPath $out -Destination $copy -Force
            Write-Output $copy
        }
    } finally {
        $graphics.Dispose(); $bitmap.Dispose()
        $inkBrush.Dispose(); $mutedBrush.Dispose(); $gridPen.Dispose(); $borderPen.Dispose()
    }
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
# Alphabetical ordering matches prior collections. Part 1 contains solvers 1-42
# (seven per panel); part 2 contains solvers 43-78 (six per panel).
try {
    foreach ($part in @(1, 2)) {
        $pageStart = if ($part -eq 1) { 0 } else { 42 }
        $solversPerPanel = if ($part -eq 1) { 7 } else { 6 }
        Draw-Figure 'ExactHits' "$hitName hits vs time" 'Reference hits out of 100 planned seeds' "${VariableCount}_${DensityCategory}_exact_hits_SM86_part${part}.png" $part $pageStart $solversPerPanel
        Draw-Figure 'WithinHits' "Seeds reaching at least 99% of $qualityReferenceName quality vs time" $withinUnit "${VariableCount}_${DensityCategory}_hits_within_1pct_SM86_part${part}.png" $part $pageStart $solversPerPanel
        Draw-Figure 'GapPercent' "Mean percentage gap to $referenceName vs time" '0% = reference; dashed line = 1% gap; lower is better' "${VariableCount}_${DensityCategory}_gap_to_reference_percent_SM86_part${part}.png" $part $pageStart $solversPerPanel
    }
} finally {
    $titleFont.Dispose(); $panelFont.Dispose(); $axisFont.Dispose()
    $legendFont.Dispose(); $subtitleFont.Dispose()
    $formatCenter.Dispose(); $formatRight.Dispose()
}
