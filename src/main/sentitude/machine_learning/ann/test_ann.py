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
    print "=" * 20, "Test Results", "=" * 20
    for each_file in test_files:
        print each_file.ljust(10) + ": " + network.get_emotion(test_data_set_path + each_file)
    print "=" * 54

    print "Now recorded: ", network.get_emotion("/home/venkatesh/Desktop/recorded.wav")
    print "Now recorded: ", network.get_emotion("/home/venkatesh/Desktop/recorded_sad.wav")


if __name__ == '__main__':
    main()
