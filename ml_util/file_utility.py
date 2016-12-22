__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

def array_from_csv_row(row, delimiter=",", next_line="\n", surrounding="\""):
    return row.replace(next_line, "").replace(surrounding, "").split(delimiter)