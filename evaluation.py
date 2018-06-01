import argparse
import numpy as np
from scipy import ndimage
import h5py

class Clefts:

    def __init__(self, test, truth):

        test_clefts = test
        truth_clefts = truth

        self.resolution=(40.0, 4.0, 4.0)
        self.truth_clefts_invalid = (truth_clefts == 0)

        self.test_clefts_mask = np.logical_or(test_clefts == 0, self.truth_clefts_invalid)
        self.truth_clefts_mask = np.logical_or(truth_clefts == 0, self.truth_clefts_invalid)
	
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=self.resolution)
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=self.resolution)

    def count_false_positives(self, threshold = 200):

        mask1 = np.invert(self.test_clefts_mask)
        mask2 = self.truth_clefts_edt > threshold
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(self, threshold = 200):

        mask1 = np.invert(self.truth_clefts_mask)
        mask2 = self.test_clefts_edt > threshold
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size

    def acc_false_positives(self):

        mask = np.invert(self.test_clefts_mask)
        false_positives = self.truth_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_positives),
            'std': np.std(false_positives),
            'max': np.amax(false_positives),
            'count': false_positives.size,
            'median': np.median(false_positives)}
        return stats

    def acc_false_negatives(self):

        mask = np.invert(self.truth_clefts_mask)
        false_negatives = self.test_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_negatives),
            'std': np.std(false_negatives),
            'max': np.amax(false_negatives),
            'count': false_negatives.size,
            'median': np.median(false_negatives)}
        return stats

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-p','--prediction',  type=str, help='prediction path')
    parser.add_argument('-g','--groundtruth', type=str, help='groundtruth path')
    args = parser.parse_args()
    return args                    

def main():
    args = get_args()

    print('0. load data')
    test = np.array(h5py.File(args.prediction, 'r')['main'])
    test = test[14:-14, 200:-200, 200:-200]
    truth = np.array(h5py.File(args.groundtruth, 'r')['main'])
    truth = truth[14:-14, 200:-200, 200:-200]
    assert (test.shape == truth.shape)
    print('volume shape:', test.shape)

    print('1. start evaluation')

    clefts_evaluation = Clefts(test, truth)

    false_positive_count = clefts_evaluation.count_false_positives()
    false_negative_count = clefts_evaluation.count_false_negatives()

    false_positive_stats = clefts_evaluation.acc_false_positives()
    false_negative_stats = clefts_evaluation.acc_false_negatives()

    print('Clefts')
    print('======')

    print('\tfalse positives: ' + str(false_positive_count))
    print('\tfalse negatives: ' + str(false_negative_count))

    print('\tdistance to ground truth: ' + str(false_positive_stats))
    print('\tdistance to proposal    : ' + str(false_negative_stats))

if __name__ == "__main__":
    main()    