ifeq ($(MAKELEVEL),0)
	CPREPROP   = -DCMDL
	MEXPREPROP = ""
	DBG   	   = -Wextra -Wall -ansi
	#DBG = -fprofile-arcs -ftest-coverage -g -pg
	DBG   	   = -pg -g -Wextra -Wall -ansi
	OPT   	   = -O2
	
	MATLABPATH := $(shell matlab -e | grep MATLAB= | sed "s/MATLAB=//")
	
	INCDIR = -I$(MATLABPATH)/extern/include -I../
	LIBDIR = -L$(MATLABPATH)/bin/glnx64
	
	MEXID    = $(shell matlab -e | grep ARCH= | sed "s/ARCH=//" | sed "s/glnx86/mexglx/" | sed "s/glnxa64/mexa64/")
	MEX      = mex
	MEXFLAGS = $(MEXPREPROP) $(INCDIR) CXXOPTIMFLAGS="$(OPT) -ffast-math" LDCXXOPTIMFLAGS="$(OPT) -ffast-math" LDOPTIMFLAGS="$(OPT) -ffast-math" $(FLAGS)
endif

all: hogf.$(MEXID)
mex: hogf.$(MEXID)

hogf.$(MEXID): hogf.cpp ../oldMatlabFix.h 
	$(MEX) $(MEXFLAGS) hogf.cpp

clean:
	rm -f hogf.mex* *.o
