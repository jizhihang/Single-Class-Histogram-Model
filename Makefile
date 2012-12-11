export CPREPROP   = -DCMDL
export MEXPREPROP = ""
export DBG   	   = -Wextra -Wall -ansi
export DBG = -fprofile-arcs -ftest-coverage -g -pg
#export DBG   	   = -pg -g -Wextra -Wall -ansi
export OPT   	   = -O2 

export MATLABPATH := $(shell matlab -e | grep MATLAB= | sed "s/MATLAB=//")

export ARCH = $(shell matlab -e | grep ARCH= | sed "s/ARCH=//")
export INCDIR = -I$(MATLABPATH)/extern/include -I../
export LIBDIR = -L$(MATLABPATH)/bin/$(ARCH)
export CPP    = g++-4.1
export CFLAGS = $(CPREPROP) $(DBG) $(OPT) $(FLAGS)
export LINK   = $(INCDIR) $(LIBDIR) -lmat -lmx -lut -licudata -licuuc -licui18n -licuio -lhdf5 -Wl,-R,$(MATLABPATH)/bin/$(ARCH)

export MEXID    = $(shell matlab -e | grep ARCH= | sed "s/ARCH=//" | sed "s/glnx86/mexglx/" | sed "s/glnxa64/mexa64/")
export MEX      = mex
export MEXFLAGS = $(MEXPREPROP) $(INCDIR) CXXOPTIMFLAGS="$(OPT) -ffast-math" LDCXXOPTIMFLAGS="$(OPT) -ffast-math" LDOPTIMFLAGS="$(OPT) -ffast-math" $(FLAGS)

##################
SUBDIRS = hog_code randomForest

all:
	for dir in $(SUBDIRS); do cd $$dir; $(MAKE) all; cd ..; done

mex:
	for dir in $(SUBDIRS); do cd $$dir; $(MAKE) mex; cd ..; done

clean:
	for dir in $(SUBDIRS); do cd $$dir; $(MAKE) clean; cd ..; done