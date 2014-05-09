__author__ = 'venkatesh'
import os
import sys
import random
from ann import Ann


def main():
    test_data_set_path = '../../../Dataset/TestData/'
    test_files = os.listdir(test_data_set_path)
    test_files = random.sample(test_files, 10)

    network = Ann()
    for each_file in test_files:
        print each_file + ": " + network.get_emotion(test_data_set_path + each_file)

if __name__ == '__main__':
    main()
