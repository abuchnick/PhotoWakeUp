import sys
import os
old_path = sys.path.copy()
sys.path.insert(1, os.path.abspath(r'.\lib\smplify-x\smplifyx'))
sys.path.insert(1, os.path.abspath(r'.\lib\smplify-x'))
from smplifyx import fitting, data_parser, fit_single_frame
import camera
import utils
import cmd_parser
import render_pkl
sys.path = old_path