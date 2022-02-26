import sys
old_path = sys.path.copy()
sys.path.insert(1, r'C:\Users\yuval\Documents\C_HW\project\AVR\lib\smplify-x\smplifyx')
sys.path.insert(1, r'C:\Users\yuval\Documents\C_HW\project\AVR\lib\smplify-x')
from smplifyx import fitting, data_parser, fit_single_frame
import camera
import utils
import cmd_parser
import render_pkl
sys.path = old_path