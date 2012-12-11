#This is part of the random Forest code
#
#Florian Schroff (schroff@robots.ox.ac.uk)
#Engineering Departement 
#University of Oxford, UK
#Copyright (c) 2008
#All Rights Reserved.
#
# exported from gitId-23lkjlajf2
#
#Use and modify all you like, but do NOT redistribute. No warranty is expressed or implied. No liability or responsibility is assumed.
#---------------------------------------
#

ifeq ($(MAKELEVEL),0)
	CPREPROP   = -DCMDL
	MEXPREPROP = ""
	DBG   	   = -Wextra -Wall -ansi
	#DBG = -fprofile-arcs -ftest-coverage -g -pg
	DBG   	   = -pg -g -Wextra -Wall -ansi
	OPT   	   = -O2
	
	MATLABPATH := $(shell matlab -e | grep MATLAB= | sed "s/MATLAB=//")
	
	ARCH = $(shell matlab -e | grep ARCH= | sed "s/ARCH=//")
	INCDIR = -I$(MATLABPATH)/extern/include -I../
	LIBDIR = -L$(MATLABPATH)/bin/$(ARCH)
	CPP    = g++-4.1
	CFLAGS = $(CPREPROP) $(DBG) $(OPT) $(FLAGS)
	LINK   = $(INCDIR) $(LIBDIR) -lmat -lmx -lut -licudata -licuuc -licui18n -licuio -lhdf5 -Wl,-R,$(MATLABPATH)/bin/$(ARCH)
	
	MEXID    = $(shell matlab -e | grep ARCH= | sed "s/ARCH=//" | sed "s/glnx86/mexglx/" | sed "s/glnxa64/mexa64/")
	MEX      = mex
	MEXFLAGS = $(MEXPREPROP) $(INCDIR) CXXOPTIMFLAGS="$(OPT) -ffast-math" LDCXXOPTIMFLAGS="$(OPT) -ffast-math" LDOPTIMFLAGS="$(OPT) -ffast-math" $(FLAGS)
endif


##################
all: precomputeResultsFromPixelDiff precomputeResultsFromPixelDiff.$(MEXID) pdDecTree pdDecTree.$(MEXID) rfClassify rfClassify.$(MEXID)
mex: pdDecTree.$(MEXID) precomputeResultsFromPixelDiff.$(MEXID) rfClassify.$(MEXID) integral_image.$(MEXID)

precomputeResultsFromPixelDiff: precomputeResultsFromPixelDiff.cxx computeDiffs.h Makefile ../Makefile
	$(CPP) $(CFLAGS) -o precomputeResultsFromPixelDiff precomputeResultsFromPixelDiff.cxx $(LINK)

precomputeResultsFromPixelDiff.$(MEXID): precomputeResultsFromPixelDiff.cxx computeDiffs.h Makefile ../Makefile
	$(MEX) $(MEXFLAGS) precomputeResultsFromPixelDiff.cxx

pdDecTree: pdDecTree.cxx computeDiffs.h Makefile ../Makefile
	$(CPP) $(CFLAGS) -o pdDecTree pdDecTree.cxx $(LINK)

pdDecTree.$(MEXID): pdDecTree.cxx computeDiffs.h Makefile ../Makefile
	$(MEX) $(MEXFLAGS) pdDecTree.cxx

rfClassify: rfClassify.cxx computeDiffs.h Makefile ../Makefile
	$(CPP) $(CFLAGS) -o rfClassify rfClassify.cxx $(LINK)

rfClassify.$(MEXID): rfClassify.cxx computeDiffs.h Makefile ../Makefile
	$(MEX) $(MEXFLAGS) rfClassify.cxx

integral_image.$(MEXID): integral_image.cxx Makefile ../Makefile
	$(MEX) $(MEXFLAGS) integral_image.cxx

clean:
	rm -f pdDecTree pdDecTree.mex* *.o precomputeResultsFromPixelDiff precomputeResultsFromPixelDiff.mex* rfClassify rfClassify.mex* integral_image.mex*
