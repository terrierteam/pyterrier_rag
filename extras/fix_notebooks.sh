fix_nb() {
    filename="$1"
    jq 'del(.metadata.widgets)'  $filename > $filename.temp
    mv $filename.temp $filename
}