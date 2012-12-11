/*%
%Florian Schroff (schroff@robots.ox.ac.uk)
%Engineering Departement 
%University of Oxford, UK
%
%Copyright (c) 2009, Florian Schroff
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
%
%    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
%    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
%    * Neither the name of the University of Oxford nor Microsoft Ltd. nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%
%Please cite the following publication(s) in published work that used or was inspired by this code/work:
%
%- Schroff, F. , Criminisi, A. and Zisserman, A.: Object Class Segmentation using Random Forests, Proceedings of the British Machine Vision Conference (2008) 
%
%---------------------------------------
%
* Input image can be any image MxNxC (uint8 if normal image, or int32 or double if integral image)
Pixeldiffs need to be int16. and 10xN: pixeldiff= [x1 y1 x2 y2 support1x support1y support2x support2y channel1 channel2; ...]'

(x1,y1) upper left corner of first diffregion, (x2,y2) respectively
supportx and supporty define the extend of diffregion
*
*/

#ifndef CMDL
    #include <mex.h>
#endif

#include <math.h>
using namespace std;

const int ninputs = 8;
//const char* inputs[]={"intI","feats","imageType","imageScale","nrNonZeroChannels","hists","sx","sy"};
const char* inputs[]={"intI","ox","oy","feats","imageType","imageScale","nrNonZeroChannels","hists"};

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <limits>

#include "computeDiffs.h"

#ifdef CMDL
    #include "readmat.h"
#else
    #include <mex.h>
#endif

#include "oldMatlabFix.h"

void precomputeResultsFromPixelDiff(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Check the inputs and outputs
	if ((nrhs < 8) || (nrhs > 10) || (nrhs==9))
		mexErrMsgTxt("precomputeResultsFromPixelDiff takes 8 or 10 inputs.");
	if (nlhs != 1)
		mexErrMsgTxt("Only one output argument allowed.");
    const mwSize* dims=(const mwSize*) mxGetDimensions(prhs[0]);
    double *ox = (double*) mxGetData(prhs[1]);
    const int osizex = (const int) *ox;
    double *oy = (double*) mxGetData(prhs[2]);
    const int osizey = (const int) *oy;

	// Create output arrays and get pointers
    if ((!mxIsClass(prhs[3],"int16") || (mxGetM(prhs[3])!=10)))
		mexErrMsgTxt("Pixeldiffs need to be int16. and 10xN: pixeldiff= [x1 y1 x2 y2 support1x support1y support2x support2y channel1 channel2; ...]'");
        
	mxClassID class_x = mxGetClassID(prhs[0]);
	void *I; 
	I = mxGetData(prhs[0]); //input image

    if ((nrhs>7) && (!mxIsClass(prhs[7],"double")))
		mexErrMsgTxt("Patch or single-class histograms need to be double.");

    // patch oder single-class histograms used in treefunction for comparison
	double *hists = NULL;
    unsigned int nrhistograms = 0;
    unsigned int nrtextons = 0;
    if (nrhs>7) {
	    hists = (double*) mxGetData(prhs[7]); 
        nrhistograms = mxGetN(prhs[7]);
        nrtextons = mxGetM(prhs[7]);
    }

	const int sizex = dims[0];
	const int sizey = dims[1];
	//const int nchannels = dims[2];
	int16_t *F = (int16_t*) mxGetData(prhs[3]); //pixel differences (functions): pixeldiff= [x1 y1 x2 y2 support channel; ...]
    const int npixdiffs = mxGetN(prhs[3]);
    int32_t *sx=NULL;
    int32_t *sy=NULL;
	int *odims;
    double *it = (double*) mxGetData(prhs[4]);
    unsigned char imageType = (unsigned char) *it;
    double *is = (double*) mxGetData(prhs[5]);
    unsigned char imageScale = (unsigned char) *is;
    double *ii = (double*) mxGetData(prhs[6]);
    unsigned int nrNonZeroCn = (unsigned int) *ii;
//    mexPrintf("imageType: %d; nrNonZeroChannels: %d\n",imageType,nrNonZeroCn);
    if (nrhs==10) {
    	if ((!mxIsClass(prhs[8],"int32")) || (!mxIsClass(prhs[9],"int32")))
			mexErrMsgTxt("sx and sy need to be int32.");
        sx = (int32_t*) mxGetData(prhs[8]); sy = (int32_t*) mxGetData(prhs[9]);
        if (mxGetNumberOfElements(prhs[9])!=mxGetNumberOfElements(prhs[8]))
		    mexErrMsgTxt("Elements in sx and sy need to be the same (arg 6 and 7).");
	    int odims_[] = {1,mxGetNumberOfElements(prhs[9]),npixdiffs};
        odims = odims_;
    } else {
        int odims_[] = {osizex,osizey,npixdiffs};
        odims = odims_;
    }
    //printf("Output dimensions: %dx%dx%d\n",odims[0],odims[1],odims[2]);
    void *X;
    unsigned int *nodeDiffs=NULL;
	switch (class_x) {
		case mxDOUBLE_CLASS:
            //printf("Treat input image as integral image!\n");
	        plhs[0] = mxCreateNumericArray(3, (const mwSize*) odims, mxDOUBLE_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            computeDiffs((double *) X,F,nodeDiffs,(double*) I,npixdiffs,sizex,sizey,sx,sy,osizex,osizey,odims,imageType,imageScale, nrNonZeroCn,hists, nrtextons, nrhistograms,(double*) NULL, (double*) NULL);
			break;
		case mxINT32_CLASS:
            //printf("Treat input image as integral image!\n");
	        plhs[0] = mxCreateNumericArray(3, (const mwSize*) odims, mxINT32_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            computeDiffs((int32_t*) X,F,nodeDiffs,(int32_t*) I,npixdiffs,sizex,sizey,sx,sy,osizex,osizey,odims,imageType,imageScale, nrNonZeroCn,hists, nrtextons, nrhistograms,(int32_t*) NULL,(int32_t*) NULL);
			break;
		case mxUINT8_CLASS:
            //printf("Treat input image as normal image!\n");
	        plhs[0] = mxCreateNumericArray(3, (const mwSize*) odims, mxINT32_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            computeDiffs((int32_t*) X,F,nodeDiffs,(uint8_t*) I,npixdiffs,sizex,sizey,sx,sy,osizex,osizey,odims,imageType,imageScale, nrNonZeroCn,hists, nrtextons, nrhistograms,(int32_t*) NULL,(int32_t*) NULL);
			break;
		default:
		    mexErrMsgTxt("Only one 3dim input image allowed of types double, int32 or uint8.");
			break;
	}
}

#ifndef CMDL

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    precomputeResultsFromPixelDiff(nlhs, plhs, nrhs, prhs);

}

#else

int main(int argc, char **argv) {
    int ndir;
    int nrhs=ninputs;
    mxArray *prhs[nrhs];
    int retval = readmat(argc, argv, prhs, ninputs, inputs, &ndir);
    if (retval) return(retval);

    /* run as if started from matlab */
    mexPrintf("\nNow run the mex part...\n");
    int nlhs=1;
    mxArray *plhs[nlhs];
    precomputeResultsFromPixelDiff(nlhs, plhs, nrhs, (const mxArray**) prhs);

    for (int i=0; i<ndir; i++) {
        mxDestroyArray(prhs[i]);
    }
    return(EXIT_SUCCESS);
}

#endif
