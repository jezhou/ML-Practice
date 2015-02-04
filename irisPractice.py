import numpy as np 
import pandas as pd

from sklearn_manage_datasets import get_digits, create_digits
from sklearn_graph import create_line_graph

def main():

	create_digits()
	create_line_graph(get_digits());

main();