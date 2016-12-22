def array_from_csv_row(row, delimiter=",", next_line="\n", surrounding="\""):
    return row.replace(next_line, "").replace(surrounding, "").split(delimiter)