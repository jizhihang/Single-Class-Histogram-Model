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
*---------------------------------------
*/

#ifndef DEBUG
    #define DEBUG 2
#endif

const int ninputs = 7;
const char* inputs[]={"forest","I","sizex","sizey","imageType","imageScale","nrNonZeroChannels"};

//#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <limits>
#include <math.h>

#include "computeDiffs.h"

#ifdef CMDL
    #include "readmat.h"
#else
    #include <mex.h>
#endif

#include "oldMatlabFix.h"

int inline add(double * a1, double * array, int N) {
    for (int i = 0; i < N; i++) {
        a1[i] += array[i];
        //mexPrintf("%f (%f)\n",a1[i],array[i]);
    }
    return 0;
} 

void rfClassify(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if ((nlhs<1) || (nrhs<7)) 
        mexErrMsgTxt("call [classification[,clusternr]]=rfClassify(forest,intI,size(I,1),size(I,2),imageType,imageScale,nrNonZeroChannels);");

    const int PintI = 1;
    const int Pforeststruct = 0;
    const int osizex = (const int) *((double*) mxGetData(prhs[2]));
    const int osizey = (const int) *((double*) mxGetData(prhs[3]));
    const int npixels = osizex *osizey;

    int forestsize = mxGetN(prhs[Pforeststruct]);
    int32_t **forest = new int32_t*[forestsize];
    int16_t **feats = new int16_t*[forestsize];
    double **posteriorL = new double*[forestsize];
    double **thresholds= new double*[forestsize];

    // patch oder single-class histograms used in treefunction for comparison
	double **hists = new double*[forestsize];
    unsigned int *nrhistograms = new unsigned int[forestsize];
    unsigned int *nrtextons = new unsigned int[forestsize];

    const unsigned int treedim=3;
    int nclasses = 0;
    for (int t=0; t<forestsize; t++) {
        if (DEBUG>=40) mexPrintf("Read ff=%d (%d)\n",t,forestsize);

        mxArray *ptreerep = mxGetField(mxGetCell(prhs[Pforeststruct],t),0,"treerep");
        if (ptreerep==NULL) mexErrMsgTxt("forest.treerep does not exist");
        if ((mxGetM(ptreerep)!=treedim) || !mxIsClass(ptreerep,"int32"))
	    	mexErrMsgTxt("forest is not cellarray of 3xN array of type int32.");
        forest[t] = (int32_t*) mxGetData(ptreerep);

        mxArray *pfeats = mxGetField(mxGetCell(prhs[Pforeststruct],t),0,"feats");
        if (pfeats==NULL) mexErrMsgTxt("forest.feats does not exist");
        if ((mxGetM(pfeats)!=10) || !mxIsClass(pfeats,"int16"))
	    	mexErrMsgTxt("feats is not cellarray of 10xN array of type int16.");
        feats[t] = (int16_t*) mxGetData(pfeats);

        mxArray *pthresholds = mxGetField(mxGetCell(prhs[Pforeststruct],t),0,"thresholds");
        if (pthresholds==NULL) mexErrMsgTxt("forest.thresholds does not exist");
        if ((mxGetM(pthresholds)!=1) || !mxIsClass(pthresholds,"double"))
	    	mexErrMsgTxt("threshold is not cellarray of 1xN array of type double.");
        thresholds[t] = (double*) mxGetData(pthresholds);

        mxArray *pposteriorL = mxGetField(mxGetCell(prhs[Pforeststruct],t),0,"posteriorL");
        if (pposteriorL==NULL) mexErrMsgTxt("forest.posteriorL does not exist");
        if (!mxIsClass(pposteriorL,"double"))
	    	mexErrMsgTxt("leaf posteriors array is not of type double.");
        nclasses = mxGetM(pposteriorL);
        posteriorL[t] = (double*) mxGetData(pposteriorL);

        mxArray *phists = mxGetField(mxGetCell(prhs[Pforeststruct],t),0,"hists");
        if (phists==NULL) mexErrMsgTxt("forest.hists does not exist");
	    hists[t] = (double*) mxGetData(phists); 
        nrhistograms[t] = mxGetN(phists);
        nrtextons[t] = mxGetM(phists);
    }

    if (!mxIsCell(prhs[PintI]) || mxGetM(prhs[PintI])<1 || mxGetN(prhs[PintI])<1 || mxGetCell(prhs[PintI],0)==NULL)
	    mexErrMsgTxt("intI is not 1xN cell array of integral images.");

    const int nrchanneltypes=(int) mxGetN(prhs[PintI]);
    void **I = new void*[nrchanneltypes];
    int *sizex = new int[nrchanneltypes];
    int *sizey = new int[nrchanneltypes];
    mxClassID *class_x = new mxClassID[nrchanneltypes];
    for (int c=0; c<nrchanneltypes; c++) {
        if (DEBUG>=40) mexPrintf("Read ct=%d (%d)\n",c,nrchanneltypes);
        mxArray *cellI = mxGetCell(prhs[PintI],c);
        class_x[c] = mxGetClassID(cellI);
        I[c] = mxGetData(cellI);
        const mwSize* dims= (mwSize*) mxGetDimensions(cellI);
	    sizex[c] = dims[0];
	    sizey[c] = dims[1];
    }
    double *imageType = ((double*) mxGetData(prhs[4]));
    double *imageScale = ((double*) mxGetData(prhs[5]));
    double *nrNonZeroCn = ((double*) mxGetData(prhs[6]));

	const int odims[] = {1,1,1};

    long int *leafindex = new long int[forestsize];
    double *posterior= new double[nclasses];

	plhs[0] = mxCreateDoubleMatrix(nclasses, npixels, mxREAL);
	double *outputI = (double*) mxGetData(plhs[0]); //output posterior image

    if (DEBUG>=10) mexPrintf("#npixels: %d\n",npixels);
    for (int p=0; p<npixels; p++) { 
    //for (int p=35; p<36; p++) { 

        int32_t sx = p%osizex+1;
        int32_t sy = p/osizex+1;
        memset(leafindex,0,forestsize*sizeof(long int));
        memset(posterior,0,nclasses*sizeof(double));
        int leafesReached=0;

        while (leafesReached<forestsize) { 
	        
            for (int t=0; t<forestsize; t++) {

                long int leaf = leafindex[t];
                if (leaf==-1) {
                    continue; //reached leaf in this tree
                }
                if (DEBUG>=25) mexPrintf("(%d,%lu): [%d %d %d]\n",t,leaf,forest[t][treedim*leaf],forest[t][treedim*leaf+1],forest[t][treedim*leaf+2]);

                if (forest[t][treedim*leaf+1]<0) {
                    leafesReached++;
                    leafindex[t]=-1; //reached leaf node
                    int32_t leafPindex = -forest[t][treedim*leaf+1]-2;
                    if (DEBUG>=20) mexPrintf("leaf=%d\n",leafPindex);
                    add(posterior,posteriorL[t]+leafPindex*nclasses,nclasses);
                    continue;
                }
                unsigned int nodeDiffs = forest[t][treedim*leaf+1];
                int16_t *pf = feats[t]+10*nodeDiffs;
                const int16_t support1x = *(pf+4); //int16_t(floor(*(pf+4)/double(imageScale)));
                const int16_t support1y = *(pf+5); //int16_t(floor(*(pf+5)/double(imageScale)));
                const int16_t support2x = *(pf+6); //int16_t(floor(*(pf+6)/double(imageScale)));
                const int16_t support2y = *(pf+7); //int16_t(floor(*(pf+7)/double(imageScale)));
                const int16_t channel1 = *(pf+8);
                const int16_t channel2 = *(pf+9);
                const int cn1 = (channel1-1)*npixels;
                const int cn2 = (channel2-1)*npixels;

                const int sup1x = support1x-1;// >= 0 ? support1x-1 : 0;
                const int sup1y = support1y-1;// >= 0 ? support1y-1 : 0;
                const int sup2x = support2x-1;// >= 0 ? support2x-1 : 0;
                const int sup2y = support2y-1;// >= 0 ? support2y-1 : 0;

                for (int c=0; c<nrchanneltypes; c++) {
                    //printf("#channel: %d\n",c);
                    if (forest[t][treedim*leaf]==c+1) { //channeltype==0 then this is node NOT a proper decision node, i.e. leafnode or non-existent node
                        double X = 0;
	                    switch (class_x[c]) {
	                    	case mxDOUBLE_CLASS:
                                computeDiffs((double*) &X,feats[t],&nodeDiffs,(double*) I[c],1,sizex[c],sizey[c],&sx,&sy,osizex,osizey,odims,(unsigned char) imageType[c],(unsigned char) imageScale[c], (unsigned int) nrNonZeroCn[c],hists[t],nrtextons[t], nrhistograms[t],(double*) NULL, (double*) NULL);
	                    		break;
	                    	case mxINT32_CLASS:
                                computeDiffs(&X,feats[t],&nodeDiffs,(int32_t*) I[c],1,sizex[c],sizey[c],&sx,&sy,osizex,osizey,odims,(unsigned char) imageType[c],(unsigned char) imageScale[c], (unsigned int) nrNonZeroCn[c],hists[t],nrtextons[t], nrhistograms[t],(double*) NULL, (double*) NULL);
	                    		break;
	                    	case mxUINT8_CLASS:
                                computeDiffs(&X,feats[t],&nodeDiffs,(uint8_t*) I[c],1,sizex[c],sizey[c],&sx,&sy,osizex,osizey,odims,(unsigned char) imageType[c],(unsigned char) imageScale[c], (unsigned int) nrNonZeroCn[c],hists[t],nrtextons[t], nrhistograms[t],(double*) NULL, (double*) NULL);
	                    		break;
	                    	default:
	                    	    mexErrMsgTxt("Only one 3dim input image allowed of types double, int32 or uint8.");
	                    		break;
	                    }
                        
                        if (DEBUG>=30) mexPrintf("%f<%f\n",X,thresholds[t][forest[t][treedim*leaf+2]]);
                        if (X<thresholds[t][forest[t][treedim*leaf+2]]) { //compare to threshold
                            leafindex[t] = 2*leafindex[t]+1; //left child
                        } else {
                            leafindex[t] = 2*leafindex[t]+2; //right child
                        }
                    }

                }

            }
        }

        memcpy(outputI+p*nclasses,posterior,nclasses*sizeof(double));

    }//END: for (pixel in I)

    delete[] leafindex;
    delete[] posterior;
    delete[] sizex;
    delete[] sizey;
}

#ifndef CMDL

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    rfClassify(nlhs, plhs, nrhs, prhs);

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
    rfClassify(nlhs, plhs, nrhs, (const mxArray**) prhs);

    for (int i=0; i<nrhs; i++) {
        mxDestroyArray(prhs[i]);
    }
    return(EXIT_SUCCESS);
}

#endif
