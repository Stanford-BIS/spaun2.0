# This file was copied from run_spaun.py and is meant to only build SPAUN
# in order to test the preprocessing and the compute_stats functions. These
# functions and this file are used in this branch to gather stats on
# SPAUN including synapse count, transform resource count and encoding/
# decoding weights.

import os
import sys
import time
import argparse
import numpy as np

import nengo
import nengo_brainstorm_pp.preprocessing as pp

# ----- Defaults -----
def_dim = 4
#def_seq = 'A'
# def_seq = 'A0[#1]?X'
# def_seq = 'A0[#1#2#3]?XXX'
def_seq = 'A1[#1]?XXX'
# def_seq = 'A2?XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
# def_seq = 'A3[1234]?XXXX'
# def_seq = 'A3[123]?XXXX'
# def_seq = 'A3[222]?XXXX'
# def_seq = 'A3[2567589]?XXXXXXXXX'
# def_seq = 'A4[5][3]?XXXXXX'
# def_seq = 'A4[321][3]?XXXXXXX'
# def_seq = 'A4[0][9]?XXXXXXXXXXX'
# def_seq = 'A4[0][9]?XXXXXXXXXXXA3[1234321]?XXXXXXXX'
# def_seq = 'A5[123]K[3]?X'
# def_seq = 'A5[123]P[1]?X'
# def_seq = 'A6[12][2][82][2][42]?XXXXX'
# def_seq = 'A6[8812][12][8842][42][8862][62][8832]?XXXXX'
# def_seq = 'A7[1][2][3][2][3][4][3][4]?XXX'
# def_seq = 'A7[1][2][3][2]?XX'
# def_seq = 'A7[1][11][111][2][22][222][3][33]?XXXXX'
# def_seq = 'A1[1]?XXA1[22]?XX'
# def_seq = '{A1[R]?X:5}'

def_mpi_p = 128

# ----- Parse arguments -----
cur_dir = os.getcwd()
parser = argparse.ArgumentParser(description='Script for running Spaun.')
parser.add_argument(
    '-d', type=int, default=def_dim,
    help='Number of dimensions to use for the semantic pointers.')
parser.add_argument(
    '-t', type=float, default=-1,
    help=('Simulation run time in seconds. If undefined, will be estimated' +
          ' from the stimulus sequence.'))
parser.add_argument(
    '-s', type=str, default=def_seq,
    help='Stimulus sequence. Use digits to use canonical digits, prepend a ' +
         '"#" to a digit to use handwritten digits, a "[" for the open ' +
         'bracket, a "]" for the close bracket, and a "X" for each expected ' +
         'motor response. e.g. A3[1234]?XXXX or A0[#1]?X')
parser.add_argument(
    '-b', type=str, default='ref',
    help='Backend to use for Spaun. One of ["ref", "ocl", "mpi", "spinn"]')
parser.add_argument(
    '--data_dir', type=str, default=os.path.join(cur_dir, 'data'),
    help='Directory to store output data.')
parser.add_argument(
    '--noprobes', action='store_true',
    help='Supply to disable probes.')
parser.add_argument(
    '--addblanks', action='store_true',
    help=('Supply to add blanks between each character in the stimulus' +
          ' sequence.'))
parser.add_argument(
    '--present_int', type=float, default=0.15,
    help='Presentation interval of each character in the stimulus sequence.')
parser.add_argument(
    '--seed', type=int, default=-1,
    help='Random seed to use.')
parser.add_argument(
    '--showdisp', action='store_true',
    help='Supply to show graphing of probe data.')

parser.add_argument(
    '--ocl', action='store_true',
    help='Supply to use the OpenCL backend (will override -b).')
parser.add_argument(
    '--ocl_platform', type=int, default=0,
    help=('OCL Only: List index of the OpenCL platform to use. OpenCL ' +
          ' backend can be listed using "pyopencl.get_platforms()"'))
parser.add_argument(
    '--ocl_device', type=int, default=-1,
    help=('OCL Only: List index of the device on the OpenCL platform to use.' +
          ' OpenCL devices can be listed using ' +
          '"pyopencl.get_platforms()[X].get_devices()" where X is the index ' +
          'of the plaform to use.'))
parser.add_argument(
    '--ocl_profile', action='store_true',
    help='Supply to use NengoOCL profiler.')

parser.add_argument(
    '--mpi', action='store_true',
    help='Supply to use the MPI backend (will override -b).')
parser.add_argument(
    '--mpi_save', type=str, default='spaun.net',
    help=('MPI Only: Filename to use to write the generated Spaun network ' +
          'to. Defaults to "spaun.net". *Note: Final filename includes ' +
          'neuron type, dimensionality, and stimulus information.'))
parser.add_argument(
    '--mpi_p', type=int, default=def_mpi_p,
    help='MPI Only: Number of processors to use.')
parser.add_argument(
    '--mpi_p_auto', action='store_true',
    help='MPI Only: Use the automatic partitioner')
parser.add_argument(
    '--mpi_compress_save', action='store_true',
    help='Supply to compress the saved net file with gzip.')

parser.add_argument(
    '--spinn', action='store_true',
    help='Supply to use the SpiNNaker backend (will override -b).')

parser.add_argument(
    '--nengo_gui', action='store_true',
    help='Supply to use the nengo_viz vizualizer to run Spaun.')

args = parser.parse_args()

# ----- Backend Configurations -----
from _spaun.config import cfg

cfg.backend = args.b
if args.ocl:
    cfg.backend = 'ocl'
if args.mpi:
    cfg.backend = 'mpi'
if args.spinn:
    cfg.backend = 'spinn'

print "BACKEND: %s" % cfg.backend.upper()

# ----- Seeeeeeeed -----
# cfg.set_seed(1413987955)
# cfg.set_seed(1414248095)
# cfg.set_seed(1429562767)
cfg.set_seed(1465947960)
#cfg.set_seed(args.seed)
print "MODEL SEED: %i" % cfg.seed

# ----- Model Configurations -----
cfg.sp_dim = args.d
cfg.raw_seq_str = args.s
cfg.present_blanks = args.addblanks
cfg.present_interval = args.present_int
cfg.data_dir = args.data_dir

if cfg.use_mpi:
    sys.path.append('C:\\Users\\xchoo\\GitHub\\nengo_mpi')

    mpi_save = args.mpi_save.split('.')
    mpi_savename = '.'.join(mpi_save[:-1])
    mpi_saveext = mpi_save[-1]

    cfg.gen_probe_data_filename(mpi_savename)
else:
    cfg.gen_probe_data_filename()

make_probes = not args.noprobes

# ----- Check if data folder exists -----
if not(os.path.isdir(cfg.data_dir) and os.path.exists(cfg.data_dir)):
    raise RuntimeError('Data directory "%s" does not exist.' % (cfg.data_dir) +
                       ' Please ensure the correct path has been specified.')

# ----- Spaun imports -----
from _spaun.utils import run_nengo_sim
from _spaun.utils import get_total_n_neurons
from _spaun.probes import idstr, config_and_setup_probes
from _spaun.spaun_main import Spaun
from _spaun.modules import get_est_runtime

# ----- Spaun proper -----
model = Spaun()

# Uncommenting the following will perform preprocessing on the model            
# and print the Brainstorm resources consumed by the model.                     
model = pp.preprocess(model, find_io = False)                                                   
calc,info = pp.calc_core_cost(model,verbose=True)     
# ----- Display stimulus seq -----
print "STIMULUS SEQ: %s" % (str(cfg.stim_seq))

# ----- Set up probes -----
if make_probes:
    print "PROBE FILENAME: %s" % cfg.probe_data_filename
    config_and_setup_probes(model)

# ----- Neuron count debug -----
print "MODEL N_NEURONS:  %i" % (get_total_n_neurons(model))
if hasattr(model, 'vis'):
    print "- vis  n_neurons: %i" % (get_total_n_neurons(model.vis))
if hasattr(model, 'ps'):
    print "- ps   n_neurons: %i" % (get_total_n_neurons(model.ps))
if hasattr(model, 'bg'):
    print "- bg   n_neurons: %i" % (get_total_n_neurons(model.bg))
if hasattr(model, 'thal'):
    print "- thal n_neurons: %i" % (get_total_n_neurons(model.thal))
if hasattr(model, 'enc'):
    print "- enc  n_neurons: %i" % (get_total_n_neurons(model.enc))
if hasattr(model, 'mem'):
    print "- mem  n_neurons: %i" % (get_total_n_neurons(model.mem))
if hasattr(model, 'trfm'):
    print "- trfm n_neurons: %i" % (get_total_n_neurons(model.trfm))
if hasattr(model, 'dec'):
    print "- dec  n_neurons: %i" % (get_total_n_neurons(model.dec))
if hasattr(model, 'mtr'):
    print "- mtr  n_neurons: %i" % (get_total_n_neurons(model.mtr))
