Single-Class-Histogram-Model
============================

Implementing Single Class Histogram Model and Texton Model For Image Segmentation

The code is developed as a part of project in Artificial Intellegence
< developed by rohanj and mrodulv @ IIT Kanpur >

Readme for SHCM and Texton Map code 

To understand how to use these files refer to their individual readme files in corresponding folders.

files in texton folder are used to build a texton map
files in shcm folder are used to build shcm for each individual class

To run the code released by F. Schroff:-
> Download it from www.robots.ox.ac.uk/~schroff/software/RF_segment_V0.1.tar.gz
> Download MSRC-21 data set:-
  > it must contain files:
		> UID_train.txt --> list of files to be used for training
		> UID_test.txt ---> list of files to be used for testing
		> original images are stored in Images folder
		> groundtruth images are stored in GroundTruth folder

> Before Running the matlab code
	> run the make file to build mex files which will be used for computing HOG features and building a decision tree.
	> Please note the matlab code is memory intensive so you may need to set the MALLOC_CHECK_ flag to 1 each time you run the training code.
		> $ export MAOLLOC_CHECK_ = 1
	> For 64bit systems, the file libc.so.6 may not be at the expected location. So link it there.
		> $ ln -s /lib/libc.so.6 /lib64/
	> On a normal desktop do not train for more than 10 trees with max depth upto 10.

> In the matlab code you have just downloaded open example.m file
	> adjust database path as required and other parameters as well
	> first function call to metarec is for trainig and second is for testing
	> for first try you may comment the other two function calls
	> for understanding what the parameters passed to metarec mean see the file metarec.m



For any clarification you may contact:
mridulv@iitk.ac.in
rohanj@iitk.ac.in

