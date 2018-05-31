import numpy as np
import h5py
from cremi import Annotations, Volume
from cremi.io import CremiFile
from cremi.evaluation import Clefts

print "0. load data"
test = h5py.File('/n/coxfs01/zudilin/research/synapse/outputs/may31_train/volume_0.h5', 'r')['main']
test = test[14:-14, 200:-200, 200:-200]
truth = h5py.File('/n/coxfs01/vcg_connectomics/cremi/gt-syn/syn_A_v2_200.h5', 'r')['main']
truth = truth[14:-14, 200:-200, 200:-200]

file1 = CremiFile("prediction.hdf", "w")
file1.write_clefts(test)
file1.close()

file2 = CremiFile("gt.hdf", "w")
file2.write_clefts(truth)
file2.close()

print "1. start evaluation"

newtest = CremiFile('prediction.hdf', 'r')
newtruth = CremiFile('gt.hdf', 'r')

clefts_evaluation = Clefts(newtest, newtruth)

false_positive_count = clefts_evaluation.count_false_positives()
false_negative_count = clefts_evaluation.count_false_negatives()

false_positive_stats = clefts_evaluation.acc_false_positives()
false_negative_stats = clefts_evaluation.acc_false_negatives()

print "Clefts"
print "======"

print "\tfalse positives: " + str(false_positive_count)
print "\tfalse negatives: " + str(false_negative_count)

print "\tdistance to ground truth: " + str(false_positive_stats) 
print "\tdistance to proposal    : " + str(false_negative_stats)