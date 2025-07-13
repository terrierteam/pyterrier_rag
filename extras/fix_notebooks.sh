fix_nb() {
    filename="$1"
    jq '.metadata.widgets."application/vnd.jupyter.widget-state+json" += {"state": {}}' $filename > $filename.temp
    mv $filename.temp $filename
}