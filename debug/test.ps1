param (
    [string]$filename,
    [Int16]$runs
)


cls
# Function to convert bias CSV file to double array

function Convert-Predictions
{
    param (
        [double[]]$predictions,
        [double]$threshold = 0.5
    )
    # Find the maximum value in the predictions
    $max_value = $predictions | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum

    # Initialize the converted predictions array
    $converted_predictions = @()

    # Convert predictions to 1 if they are equal to the maximum value, otherwise to 0
    foreach ($prediction in $predictions)
    {
        if ($prediction -eq $max_value)
        {
            $converted_predictions += 1
        }
        else
        {
            $converted_predictions += 0
        }
    }

    return $converted_predictions
}

# Function to apply weights, biases and activation function
function Start-ApplyLayer
{
    param(
        [double[]]$input_data,
        $weights,
        [double[]]$biases,
        [string]$activation
    )
    [Double[]]$output_data = @()
    $rows = $weights.Length
    $cols = $weights[0].length

    for ($i = 0; $i -lt $cols; $i++) {
        [Double]$sum = 0
        for ($j = 0; $j -lt $rows; $j++) {
            $sum += $input_data[$j] * $weights[$j][$i]
        }
        $sum += $biases[$i]
        if ($activation -eq "relu")
        {
            $output_data += [Double]([Math]::Max([Double]0, [Double]$sum))
        }
        elseif ($activation -eq "linear")
        {
            $output_data += $sum
        }
        elseif ($activation -eq "sigmoid")
        {
            $output_data += 1 / (1 + [Math]::Exp(-$sum))
        }

    }
    return $output_data
}

# Debugging function to print arrays
function Print-DebugArray
{
    param(
        [string]$name,
        [double[]]$array
    )
    Write-Output "$($name.PadRight(17) ): $( ($array | % { $_.ToString().PadLeft(2) }) -join ', ' )"
}

function Get-ReverseBoardInput
{
    param (
        [int[]]$board
    )

    $new_board = @()

    for ($i = 0; $i -lt 27; $i += 3) {
        if ($board[$i] -eq 1)
        {
            $new_board += 0
        }
        elseif ($board[$i + 1] -eq 1)
        {
            $new_board += 1
        }
        else
        {
            $new_board += -1
        }
    }

    return $new_board
}

function Convert-BiasToDoubleArray
{
    param(
        [string]$csvFilePath
    )
    $array = @()
    $lines = Get-Content -Path $csvFilePath
    foreach ($line in $lines)
    {
        $array += [double]$line
    }
    return $array
}

# Function to convert weights CSV file to double array
function Convert-WeightsToDoubleArray
{
    param(
        [string]$csvFilePath
    )
    $array = @()
    $lines = Get-Content -Path $csvFilePath
    Set-Clipboard -Value $lines
    foreach ($line in $lines)
    {
        $weights = $line -split ';'
        $doubleWeights = $weights | ForEach-Object { [double]$_ }
        $array += ,@($doubleWeights)
    }
    return $array
}

# Load weights and biases from CSV files
$weights_layer_0 = Convert-WeightsToDoubleArray "weight_layer_0.csv"
$biases_layer_0 = Convert-BiasToDoubleArray "bias_layer_0.csv"
$weights_layer_1 = Convert-WeightsToDoubleArray "weight_layer_1.csv"
$biases_layer_1 = Convert-BiasToDoubleArray "bias_layer_1.csv"
$weights_layer_2 = Convert-WeightsToDoubleArray "weight_layer_2.csv"
$biases_layer_2 = Convert-BiasToDoubleArray "bias_layer_2.csv"
$weights_layer_3 = Convert-WeightsToDoubleArray "weight_layer_3.csv"
$biases_layer_3 = Convert-BiasToDoubleArray "bias_layer_3.csv"
$weights_layer_4 = Convert-WeightsToDoubleArray "weight_layer_4.csv"
$biases_layer_4 = Convert-BiasToDoubleArray "bias_layer_4.csv"


$dataset = Get-Content $filename | ConvertFrom-Json

$input_datas = $dataset.states
$expected_outputs = $dataset.next_states
# Input data



$correct = 0
$samples = $runs
$samples = [math]::Min($input_datas.Length, $samples)
$cinda_correct = 0
for ($i = 0; $i -lt $samples; $i++) {
    clear
    Write-Host $i/$samples
    $input_data = $input_datas[$i]
    $expected_output = $expected_outputs[$i]

    $start = Get-Date
    # Apply weights to input data
    $output_layer1 = Start-ApplyLayer $input_data $weights_layer_0 $biases_layer_0 -activation "relu"
    $output_layer2 = Start-ApplyLayer $output_layer1 $weights_layer_1 $biases_layer_1 -activation "relu"
    $output_layer3 = Start-ApplyLayer $output_layer2 $weights_layer_2 $biases_layer_2 -activation "relu"
    $output_layer4 = Start-ApplyLayer $output_layer3 $weights_layer_3 $biases_layer_3 -activation "relu"
    $output_layer = Start-ApplyLayer $output_layer4 $weights_layer_4 $biases_layer_4 -activation "linear"

    $end = Get-Date
    # Output the result
    # conmpare the result with the input data
    $converted_input = Get-ReverseBoardInput -board $input_data
    $converted_prediction = Convert-Predictions -predictions $output_layer -threshold 0.5

    if ((($converted_prediction | Measure-Object -Sum).Sum -ne 1) -or $converted_input[($converted_prediction.indexof(1))] -ne 0)
    {
        $not_valid = $true
        $cinda = $false
    }
    else
    {
        $not_valid = $false
        $cinda = $true
        $cinda_correct++
    }
    $match = $true
    For ($j = 0; $j -lt $converted_prediction.Length; $j++) {
        if ($converted_prediction[$j] -ne $expected_output[$j])
        {
            $match = $false
            break
        }
    }
    if ($match)
    {
        $correct++
    }
    if (-not $not_valid)
    {
        continue
    }

    Write-Host "---------------------------------"
    Write-Host ("Time taken       : $([Math]::Round(($end - $start).TotalMilliseconds, 2)) ms")
    Print-DebugArray "Output Layer" $output_layer
    Print-DebugArray "Input Data" $converted_input
    Print-DebugArray "Expected Output" $expected_output
    Print-DebugArray "Predictions" $converted_prediction
    Write-Host ("Not Valid        : " + $not_valid)
    Write-Host ("Match            : " + $match)
    Write-Host ("Cinda            : " + $cinda)
}
Write-Host "---------------------------------"
Write-Host "Cinda Correct    : $cinda_correct / $samples"
Write-Host "Cinda Accuracy   : $( $cinda_correct / $samples * 100 )%"
Write-Host "Correct          : $correct / $samples"
Write-Host "Accuracy         : $( $correct / $samples * 100 )%"


