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
*/

#ifndef DEBUG
    #define DEBUG 0
#endif

#define LEAFFEATS 10
#define MAXDEPTH 3
#define MINENTROPY 0
#define MINGAIN 0

//#define isnan(x) ((x) != (x))

const int ninputs = 5;
const char* inputs[]={"intI","gt","pixeldiffs","treeparams","nhists"};

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

double mean (double * array, double N) {
    double sum = 0 ;
    for (int i = 0; i < N; i++)
        sum = sum + array [i];
    return sum/N;
} // function calculating mean


double std_dev (double * array, double N) {
    double sum = 0;
    double STD_DEV = 0; // returning zero's

    for (int i = 0; i < N; i++) {
        sum = sum + array [i];
        STD_DEV = STD_DEV + pow(array [i], 2);
    }
    return sqrt ((STD_DEV/N) - (pow(sum/N,2)));
} // function calculating standard deviation


class TreeParameters {
    public:
        unsigned int maxTreeDepth;
        double minEntropy;
        double minGain;
        unsigned int nFeatsPerNode;
        bool normalizeByPrior;
        double *imageType;
        double *imageScale;
        double *nrNonZeroChannels;
        int nrsamples;
        unsigned int nclasses;
        unsigned int nrchanneltypes;
        int threshoptimize;


        TreeParameters(): maxTreeDepth(MAXDEPTH), minEntropy(MINENTROPY), minGain(MINGAIN), nFeatsPerNode(LEAFFEATS), normalizeByPrior(false) {
            nrchanneltypes = 0;
            imageType = NULL;
            imageScale= NULL;
            nrNonZeroChannels = NULL;
            threshoptimize = 0;
        }

        TreeParameters(const mxArray *params) {
            set(params);
        }

        ~TreeParameters() {
        }

        void set(const mxArray *params) {
            if ((mxGetM(params)!=1) || !mxIsCell(params))
                mexErrMsgTxt("Wrong input format for tree parameters (not a cell array).");

            int nparams = mxGetN(params);
            if (nparams>=1) nclasses = (unsigned int)         *((double*)mxGetData(mxGetCell(params,0))); else nclasses = 0;
            if (nparams>=2) maxTreeDepth = (unsigned int)     *((double*)mxGetData(mxGetCell(params,1))); else maxTreeDepth = MAXDEPTH;
            if (nparams>=3) minEntropy =                     -*((double*)mxGetData(mxGetCell(params,2))); else minEntropy = MINENTROPY;
            if (nparams>=4) minGain =                        -*((double*)mxGetData(mxGetCell(params,3))); else minGain = MINGAIN;
            if (nparams>=5) nFeatsPerNode = (unsigned int)    *((double*)mxGetData(mxGetCell(params,4))); else nFeatsPerNode = LEAFFEATS;
            if (nparams>=6) normalizeByPrior= (bool)          *((double*)mxGetData(mxGetCell(params,5))); else normalizeByPrior = false;
            if (nparams>=7) {
                nrchanneltypes = mxGetN(mxGetCell(params,6));
            }else{
                nrchanneltypes = 1;
            }
            imageType = new double[nrchanneltypes];
            if (nparams>=7) {
                imageType = (double*) mxGetData(mxGetCell(params,6)); 
            }else{
                *imageType = 0;
            }
            imageScale = new double[nrchanneltypes];
            if (nparams>=8) {
                imageScale = (double*) mxGetData(mxGetCell(params,7)); 
            }else{
                *imageScale = 0;
            }

            unsigned int nrct = 1;
            if (nparams>=9) {
                nrct = mxGetN(mxGetCell(params,8));
            }
            if (nrct != nrchanneltypes) mexErrMsgTxt("Nr channels in imageType and nrNonZeroChannels must agree!\n");
            nrNonZeroChannels = new double[nrchanneltypes];
            if (nparams>=9) {
                nrNonZeroChannels = (double*) mxGetData(mxGetCell(params,8)); 
            }else{
                *nrNonZeroChannels= 0;
            }
            if (nparams>=10) nrsamples= (int)         *((double*)mxGetData(mxGetCell(params,9))); else nrsamples= 0;
            if (nparams>=11) threshoptimize= (int)         *((double*)mxGetData(mxGetCell(params,10))); else threshoptimize= 0;

            mexPrintf("Use:\nnclasses=%d;\nmaxTreeDepth=%d;\nminEntropy=%e;\nminGain=%e;\nnFeatsPerNode=%d;\nnormalizByPrior=%d;\nnrsamples=%d;\nthreshoptimize=%d;\n\n",nclasses, maxTreeDepth,minEntropy,minGain,nFeatsPerNode,normalizeByPrior,nrsamples,threshoptimize);
            for (unsigned int i=0; i<nrchanneltypes; i++) {
                mexPrintf("imageType%d=%d; imageScale%d=%d; nrNonZerochannels=%d\n",i,(int)imageType[i],i,(int)imageScale[i],(int)nrNonZeroChannels[i]);
            }
            fflush(stdout);
        }
};

bool isParentLeaf(mxArray* leafNodes, unsigned long int currentnode) {
    //mexPrintf("leafnodes[%d]=%p\n",n,mxGetCell(leafNodes,n));
    unsigned long int p=(unsigned long int) floor((currentnode-2)/2);
    if (mxGetCell(leafNodes,p)!=NULL) {
        //mark both children als leave nodes so their children don't get visited again
        mxArray *tmp = mxCreateDoubleMatrix(1,1,mxREAL);
        double *p=(double*) mxGetData(tmp);
        *p = mxGetNaN();
        //printf("%lu\n",currentnode);
        mxSetCell(leafNodes,currentnode-1,tmp);
        mxArray *tmp1 = mxDuplicateArray(tmp);
        mxSetCell(leafNodes,currentnode,tmp1);
        return true;
    }
    return false;
}

unsigned long int getFeatureResponses(double *X, int32_t *SX, int32_t *SY, double *resHist, uint32_t *resHistH, int16_t *F, unsigned int *nodeDiffs, mxClassID class_x, void **images,int *imgdims,int *imgdimsGT,void **gt,unsigned int nrimgs,uint64_t *leafindx, unsigned int nFeatsPerNode, int imageType, int imageScale, unsigned long int currentnode, double **threshold, bool setleafIndx, unsigned int nrNonZeroCn, double *hists, unsigned int nrtextons, unsigned int nrhistograms, TreeParameters tps,unsigned int nrGTclasses) {


    //get total number of pixels in this node
    unsigned int offset=0;
    unsigned int *m = new unsigned int[nFeatsPerNode];
    unsigned long int totalcomppixels=0;
    double *minResponse = new double[nFeatsPerNode];
    double *maxResponse = new double[nFeatsPerNode];
    double *minResp = new double[nFeatsPerNode];
    double *maxResp = new double[nFeatsPerNode];
    for (unsigned int p=0; p<nFeatsPerNode; p++) {
        minResponse[p] = 0; //std::numeric_limits<double>::max();
        maxResponse[p] = 0; //std::numeric_limits<double>::min();
    }
    bool setThresh = false;
    int threshoptimize = (tps.threshoptimize<=0) ? 2 : tps.threshoptimize;

for (int nRun=0; nRun<2; nRun++) {
    
    if ((*threshold)==NULL) {
        *threshold = new double[tps.nFeatsPerNode*threshoptimize];
        setThresh = true;
        memset(m,0,tps.nFeatsPerNode*sizeof(unsigned int));
    } else {
        nRun=1; //if thresholds already set skip first run
    }


    offset=0;
    totalcomppixels=0;
    for (unsigned int i=0; i<nrimgs; i++) {
        
        const int sizexGT = imgdimsGT[2*i];
        const int sizeyGT = imgdimsGT[2*i+1];
        const int npixelsGT = imgdimsGT[2*i]*imgdimsGT[2*i+1];
        const int sizex = imgdims[2*i];
        const int sizey = imgdims[2*i+1];
        mxArray *I= (mxArray*) images[i];

        int32_t *sx=SX; int32_t *sy=SY;
        unsigned int comppixels=0;
        for (int l=0; l<npixelsGT; l++) {
            if (leafindx[l+offset]==currentnode) { //TODO: this is really slow, faster accounting of where data is???
                *(sx++) = int32_t (l % sizexGT)+1; //+1 to have same format as matlab input 
                *(sy++) = int32_t (l / sizexGT)+1; 
                if (nRun==0) totalcomppixels++;
                comppixels++;
            }
        }
        if (! setleafIndx) {
            if (DEBUG>=50) {
                mexPrintf("\n\nCompute %d pixels in image %d\n",comppixels,i);
            } else if (DEBUG>=3) { 
                mexPrintf("."); 
                fflush(stdout); 
            }
        }

	    int odims[] = {1,comppixels,nFeatsPerNode};

        memset(X,0,comppixels*nFeatsPerNode*sizeof(double)); //need to reset
        //sx=NULL; sy=NULL;
	    switch (class_x) {
	    	case mxDOUBLE_CLASS:
                //mexPrintf("Treat input image as integral image!\n");
                computeDiffs(X,F,nodeDiffs,(double*) I,nFeatsPerNode,sizex,sizey,SX,SY,sizexGT,sizeyGT,odims,imageType,imageScale,nrNonZeroCn,hists, nrtextons, nrhistograms, minResp, maxResp);
	    		break;
	    	case mxINT32_CLASS:
                //mexPrintf("Treat input image as integral image!\n");
                computeDiffs(X,F,nodeDiffs,(int32_t*) I,nFeatsPerNode,sizex,sizey,SX,SY,sizexGT,sizeyGT,odims,imageType,imageScale,nrNonZeroCn,hists, nrtextons, nrhistograms, minResp, maxResp);
	    		break;
	    	case mxUINT8_CLASS:
                //mexPrintf("Treat input image as normal image!\n");
                computeDiffs(X,F,nodeDiffs,(uint8_t*) I,nFeatsPerNode,sizex,sizey,SX,SY,sizexGT,sizeyGT,odims,imageType,imageScale,nrNonZeroCn,hists, nrtextons, nrhistograms, minResp, maxResp);
	    		break;
	    	default:
	    	    mexErrMsgTxt("Only one 3dim input image allowed of types double, int32 or uint8.");
	    		break;
	    }



        //sample threshold from all thresholds (Selection Sampling, see Knuth The art of computer programming Vol.2)
        if (nRun==0) {
            for (unsigned int p=0; p<nFeatsPerNode; p++) {
                minResponse[p] = (minResp[p] < minResponse[p]) ? minResp[p] : minResponse[p];
                maxResponse[p] = (maxResp[p] > maxResponse[p]) ? maxResp[p] : maxResponse[p];
            }
            if (setThresh) {
                if (tps.threshoptimize==0) { //select threshold randomly
                    for (unsigned int p=0; p<nFeatsPerNode; p++) {
                            double u=((double) random())/RAND_MAX;
                            (*threshold)[p]=minResponse[p]+(maxResponse[p]-minResponse[p])*u;
                            if (DEBUG>=40) mexPrintf("th=%f\n",(*threshold)[p]);
                    }
                }
                if (tps.threshoptimize==-1) { //use threshold zero
                    for (unsigned int p=0; p<nFeatsPerNode; p++) {
                        (*threshold)[p]=0;
                    }
                }
            }
        }
            
        if (nRun==1) {
        /*** compute histogram ***/
            unsigned char *GT = (unsigned char*) gt[i];
            for (unsigned int oc=0; oc<nrGTclasses; oc++) {
                sx=SX; sy=SY;
                for (unsigned int c=0; c<comppixels; c++) {
                    unsigned int pixind = *sx-1+(*sy-1)*sizexGT;
                    sx++; sy++;
                    unsigned char cl = GT[pixind+oc*npixelsGT];
                    if (tps.nclasses!=0 && cl>tps.nclasses) {
                        //printf("%d: %d %d\n",cl,sx,sy); //can happen due to oc, despite being pos training pixels they might not be in all layers/oc
                        continue;
                    }

                    resHist[cl-1]++;
                    totalcomppixels++; //needs to be counted here due to the multiple layers, oc, and the continue in the if-statement above
                    for (unsigned int p=0; p<nFeatsPerNode; p++) {
                        
                        if (setThresh && tps.threshoptimize>0) {
                            double v = X[c+comppixels*p];
                            for (int i=0; i<tps.threshoptimize; i++) {
                                if (((i==0) || (v >= (*threshold)[p+(i-1)*tps.nFeatsPerNode])) && ((i+1==tps.threshoptimize) || (v < (*threshold)[p+i*tps.nFeatsPerNode]))) { 
                                    resHistH[cl-1+tps.nclasses*p+tps.nclasses*i*tps.nFeatsPerNode]++;
                                    //cct++;
                                }
                            }
                        } else {
                            if (X[c+comppixels*p] < (*threshold)[p]) { //left node
                                resHistH[cl-1+tps.nclasses*p]++;
                                if (setleafIndx) leafindx[pixind+offset]=2*(currentnode-1)+2; //2*i+1 (and adjust for root node being node=1 instead of 0)
                            } else { //right node
                                resHistH[cl-1+tps.nclasses*p+tps.nclasses*tps.nFeatsPerNode]++;
                                if (setleafIndx) leafindx[pixind+offset]=2*(currentnode-1)+3; //2*i+2 (and adjust for root node being node=1 instead of 0)
                            }
                        }
                     }
                 }
            }
        }

        offset +=npixelsGT; //offset adjusting for sequential load of images

	} //END: for (int i=0; i<nrimgs; i++) 

    if ((nRun==0) && (setThresh && tps.threshoptimize>0)) {
        for (unsigned int p=0; p<nFeatsPerNode; p++) {
            double binsize = (maxResponse[p]-minResponse[p])/tps.threshoptimize;
            binsize = binsize==0 ? 1 : binsize;
            double oldmin = minResponse[p];
            double oldmax = maxResponse[p];
            minResponse[p] = round(minResponse[p]/binsize)*binsize;
            maxResponse[p] = binsize*tps.threshoptimize+minResponse[p];

            for (int i=0; i<tps.threshoptimize; i++) {
                (*threshold)[p+i*tps.nFeatsPerNode] = minResponse[p]+(i+1)*binsize;
                if (DEBUG>=40) mexPrintf("\nbin %d = %e;\n",i,(*threshold)[p+i*tps.nFeatsPerNode]);
            }
            
            if ((DEBUG>=40) && (oldmin!=minResponse[p])) mexPrintf("\n%d minR: %f!=%f; maxR: %f!=%f; bin: %f\n",p,oldmin,minResponse[p],oldmax,maxResponse[p],binsize);
        }
    }


}
    //compute cumulative histogram
    for (unsigned int p=0; p<nFeatsPerNode; p++) {
    for (unsigned int c=0; c<tps.nclasses; c++) { 
    for (int i=1; i<threshoptimize; i++) {
        resHistH[c+tps.nclasses*p+i*tps.nclasses*tps.nFeatsPerNode] += resHistH[c+tps.nclasses*p+(i-1)*tps.nclasses*tps.nFeatsPerNode];
    }}}
    delete[] minResponse;
    delete[] maxResponse;
    delete[] minResp;
    delete[] maxResp;
    delete[] m;
    return totalcomppixels;
}

unsigned int getSplitEntropies(double *resHist, uint32_t *resHistH, unsigned int nFeatsPerNode, double totalcomppixels, double *gain, double *entropy, double *bestld, double *bestrd, TreeParameters tps, double **threshold, double *classPrior) {
    int bestfeat = -1;
    unsigned int threshoptimize;
    threshoptimize = (threshold!=NULL && tps.threshoptimize>0) ? tps.threshoptimize : 2;
    double x=0;
    double bestgain=1; //info_gain<=0
    if (DEBUG>=5) {
        mexPrintf("\n");
    }
    if (tps.normalizeByPrior) {
        totalcomppixels=0;
        for (unsigned int c=0; c<tps.nclasses; c++) {
            double a = resHist[c]>0 ? resHist[c]/classPrior[c] : 0; //"virtual" pixels
            totalcomppixels += a;
        }
        //x=-1000000000; //set to some constant value
    } //else {
    for (unsigned int c=0; c<tps.nclasses; c++) {
        double a = resHist[c];
        if ((a>0) && (tps.normalizeByPrior)) a /=classPrior[c];
        a = a/totalcomppixels; //normalize to compute entropy
        if (a>0) x += a * log(a);
        if (DEBUG>=30) mexPrintf("%d: total=%f; %f \n",c,totalcomppixels,a);
    }
    //}
    if (DEBUG>=20) {
        mexPrintf("\nnFeatsPerNode=%d; nclasses=%d; threshoptimize=%d\n",nFeatsPerNode,tps.nclasses,threshoptimize);
    }
    double ld=0,rd=0;
    for (unsigned int p=0; p<nFeatsPerNode; p++) {
        for (unsigned int bin=0; bin<threshoptimize-1; bin++) {

            double xr=0,xl=0;
            ld=0; rd=0;

            // loop over all threshold bins to find the best
            for (unsigned int c=0; c<tps.nclasses; c++) {
                double a=0;
                a = resHistH[c+tps.nclasses*p+bin*tps.nFeatsPerNode*tps.nclasses]; //use cumulative histogram
                if (tps.normalizeByPrior && a>0) a /= classPrior[c];
                ld += a;
                
                a = 0;
                a = double(resHistH[c+tps.nclasses*p+(threshoptimize-1)*tps.nFeatsPerNode*tps.nclasses]) - double(resHistH[c+tps.nclasses*p+bin*tps.nFeatsPerNode*tps.nclasses]); //use cumulative histogram
                if (tps.normalizeByPrior && a>0) a /= classPrior[c];
                rd += a;
            }

            if (ld==0) xl=0; else {
                for (unsigned int c=0; c<tps.nclasses; c++) {
                    double a=0;
                    a = resHistH[c+tps.nclasses*p+bin*tps.nFeatsPerNode*tps.nclasses]; //use cumulative histogram
                    if (tps.normalizeByPrior) a /= classPrior[c];
                    a = a/ld; //normalize to compute entropy
                    if (a>0) xl += a * log(a);
                }
            }
            if (rd==0) xr=0; else {
                for (unsigned int c=0; c<tps.nclasses; c++) {
                    double a = 0;
                    a = double(resHistH[c+tps.nclasses*p+(threshoptimize-1)*tps.nFeatsPerNode*tps.nclasses]) - double(resHistH[c+tps.nclasses*p+bin*tps.nFeatsPerNode*tps.nclasses]); //use cumulative histogram
                    if (tps.normalizeByPrior) a /= classPrior[c];
                    a = a/rd; //normalize to compute entropy
                    if (a>0) xr += a * log(a);
                }
            }

		    double info_gain=x-ld/totalcomppixels*xl-rd/totalcomppixels*xr; //NB entropies are negative should be -x,-xl and -xr;
            if ((DEBUG>=25) && (x==0) && (info_gain==0)) {
                mexPrintf("%2.d Current-entropy: %e: EL: %e; ER: %e; Gain: %e; ld=%f; rd=%f\n",p,x,xl,xr,info_gain,ld,rd);
            }

            if (((rd+ld)-totalcomppixels>0.0000001) || ((rd+ld)-totalcomppixels<-0.0000001)) {
                mexPrintf("%2.d Current-entropy: %e: EL: %e; ER: %e; Gain: %e; ld=%f; rd=%f; tot=%f\n",p,x,xl,xr,info_gain,ld,rd,totalcomppixels);
                mexErrMsgTxt("UUUUUPS");
            }
            if (info_gain<bestgain) {
                bestgain=info_gain;
                bestfeat=p;
                *bestld = ld;
                *bestrd = rd;
                if (threshold!=NULL && tps.threshoptimize>0) {
                    (*threshold)[p] = (*threshold)[p+bin*tps.nFeatsPerNode];
                }
            }
            if (DEBUG>=40) {
                if (threshold!=NULL) {
                    mexPrintf("Bin=%d; %2.d Current-entropy: %e: EL: %e; ER: %e; Gain: %e; ld=%d; rd=%d; thresh=%e\n",bin,p,x,xl,xr,info_gain,(unsigned int) ld,(unsigned int) rd,(*threshold)[p+bin*tps.nFeatsPerNode]);
                }else{
                    mexPrintf("Bin=%d; %2.d Current-entropy: %e: EL: %e; ER: %e; Gain: %e; ld=%d; rd=%d\n",bin,p,x,xl,xr,info_gain,(unsigned int) ld,(unsigned int) rd);
                }
            }
        }
    }
    if (DEBUG>=5) {
        if  (bestfeat<0) mexErrMsgTxt("Couldn't select feature: bestfeat<0\n");
        mexPrintf("Selected feature: %d; current-entropy: %e; Gain: %e;\n",bestfeat,x,bestgain);
    }
    *gain = bestgain;
    *entropy = x;
    return bestfeat;
}

void pdDecTree(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    if ((nlhs<3) || (nrhs<3)) {
        mexErrMsgTxt("call [tree,leaves,thresholds [,treestats,seg]]=pdDecTree(intI,gt,pixeldiffs [,treeparams, histograms]);");
    }
//these are index values to gather required parameters from prhs which is an array storing all the input parameters to 
// pDecTree from do_lpd.m file
    const int PintI = 0; 
    const int Pgt = 1;
    const int Ppixeldiffs = 2;
    const int Ptreeparams = 3;
    const int Phistograms = 4;

//fprintf(stderr,"pdectree-1\n");
    if (!mxIsCell(prhs[PintI]) || mxGetM(prhs[PintI])<1 || mxGetN(prhs[PintI])<1 || mxGetCell(prhs[PintI],0)==NULL)
		mexErrMsgTxt("First input is not 1xN cell array of integral images.");

    if (!mxIsCell(prhs[Pgt]) || (mxGetM(prhs[Pgt])!=1) || (mxGetN(prhs[Pgt])<1) || (mxGetCell(prhs[Pgt],0)==NULL) || !mxIsClass(mxGetCell(prhs[Pgt],0),"uint8"))
		mexErrMsgTxt("Second input is not 1xN cell array of groundtruth images of type uint8.");

    /*** Read and set tree parameters ***/
    // prhs = [intI , gt, pixeldiff, treeparams,nhists]
    //mxgetM mxGetN returns the dimensions of an array
    
    
    
    TreeParameters tps;
    tps.set(prhs[Ptreeparams]);
    //fprintf(stderr,"pdectree-2\n");
    const unsigned int nrimgs=mxGetM(prhs[PintI]);   // no. of images. As prhs[PintI] = intI is s.t. intI{i,ft} = i is the feat response
						     //for that particular type of feature. ft = 0 being HOG, ft = 1 being RGB etc.
    const unsigned int nrchanneltypes=mxGetN(prhs[PintI]); // no.of channel types i.e. len['hog','rgb',AC17] = 3
    mexPrintf("%d training image(s). %d channel types.\n",nrimgs,nrchanneltypes);
    if (nrimgs!=mxGetN(prhs[Pgt]))
		mexErrMsgTxt("Second input is not 1xN cell array of groundtruth images with the same number of images as the first input argument.");

    mxArray *I = mxGetCell(prhs[PintI],0);
    
    if (3!=mxGetNumberOfDimensions(I))
		mexErrMsgTxt("Input images need to be 3 dimensional MxNxC");

    const unsigned int pixdiffDim = 10;
    if ((!mxIsClass(prhs[Ppixeldiffs],"int16") || (mxGetM(prhs[Ppixeldiffs])!=pixdiffDim)))
		mexErrMsgTxt("Pixeldiffs need to be int16. and 10xN: pixeldiff= [x1 y1 x2 y2 support1x support1y support2x support2y channel1 channel2; ...]'");
		// see the structure of pixeldiff. it is a 10xN array storing just position and size of 2 rectangles and their channel
    if ((nrhs>4) && (!mxIsClass(prhs[Phistograms],"double")))
		mexErrMsgTxt("Patch or single-class histograms need to be double.");
     //fprintf(stderr,"pdectree-3\n");   
    //read in the cell array of images and extract dimensions and stuff
    void **images = new void* [nrchanneltypes*nrimgs];
    mxClassID *iClass = new mxClassID[nrchanneltypes];
    void **gt = new void* [nrimgs];
    int *imgdims = new int[nrchanneltypes*nrimgs*2];
    int *imgdimsGT = new int[nrimgs*2];
    unsigned long int nrtotalpoints=0;
    unsigned int maxpixels=0; // maximum no. of pixels in any image
    unsigned nrGTclasses = 0; //this allowes to specify nrGTclasses each pixel belongs to
    mwSize* dimsGT;
    
    fprintf(stderr,"here..\n");
    
    for (unsigned int ct=0; ct<nrchanneltypes; ct++) {
	    //fprintf(stderr,"pdectree-4\n");
        I=mxGetCell(prhs[PintI],ct*nrimgs);
        mwSize* dims= (mwSize*) mxGetDimensions(I);
	    iClass[ct] = mxGetClassID(I);
	    unsigned int nchannels = dims[2];
        for (unsigned int i=0; i<nrimgs; i++) {
		//fprintf(stderr,"pdectree-5\n");
            I=mxGetCell(prhs[PintI],i+ct*nrimgs);
            dims= (mwSize*) mxGetDimensions(I);
            imgdims[2*i+ct*2*nrimgs]=dims[0]; //number of image rows
            imgdims[2*i+1+ct*2*nrimgs]=dims[1]; //number of image columns
            mxArray *GT = mxGetCell(prhs[Pgt],i);
            dimsGT= (mwSize*) mxGetDimensions(GT);
            if (3!=mxGetNumberOfDimensions(I) || (dims[2]!=nchannels) || (iClass[ct]!=mxGetClassID(I)) || 
               (2!=mxGetNumberOfDimensions(GT) && nrGTclasses!=0 && dimsGT[2]!=nrGTclasses) || (tps.imageScale[ct]==1 && dims[0]!=dimsGT[0]) || (tps.imageScale[ct]==1 && dims[1]!=dimsGT[1]) ) {
                mexPrintf("Img#: %d\n",i);
                mexPrintf("%d\n",mxGetNumberOfDimensions(I));
                mexPrintf("%d\n",dims[0]);
                mexPrintf("%d\n",dims[1]);
                mexPrintf("%d\n",nchannels);
                mexPrintf("%d\n",mxGetNumberOfDimensions(GT));
                mexPrintf("%d\n",dimsGT[0]);
                mexPrintf("%d\n",dimsGT[1]);
	    	    mexErrMsgTxt("Not all input images (and groundtruth) have the same image type, number of channels and 3 dimensions.");
            }
            if (mxGetNumberOfDimensions(GT)==3) nrGTclasses = dimsGT[2];
            images[ct*nrimgs+i]=mxGetData(I);
            gt[i]=mxGetData(GT);
            imgdimsGT[2*i]=dimsGT[0]; //number of image rows
            imgdimsGT[2*i+1]=dimsGT[1]; //number of image columns
            nrtotalpoints += dimsGT[0]*dimsGT[1];
            const unsigned int npixels = dimsGT[0]*dimsGT[1];
            if (npixels>maxpixels) maxpixels=npixels;
        }
    }
    //fprintf(stderr,"pdectree-6\n");
    nrtotalpoints = nrtotalpoints / nrchanneltypes; //to account for multiple counting in loops above

    if (nrtotalpoints>UINT_MAX) {
        mexErrMsgTxt("Too many training points for 32bit");
    }

    // get the precomputed pool of pixel-differences: pixeldiff= [x1 y1 x2 y2 support channel; ...]
	int16_t *F = (int16_t*) mxGetData(prhs[Ppixeldiffs]); 
    unsigned int nallpixdiffs = mxGetN(prhs[Ppixeldiffs]); //total no. of pixel diffs
    nallpixdiffs /= tps.nrchanneltypes; // no.of pixel diffs per channel/feature type
    if (nallpixdiffs < tps.nFeatsPerNode) 
        mexErrMsgTxt("Oh no, can't have fewer pixeldiffs in the pool than per node...");
//fprintf(stderr,"pdectree-7\n");
    // patch oder single-class histograms used in treefunction for comparison
//getting hist. parameters
	double *hists = NULL;
    unsigned int nrhistograms = 0;
    unsigned int nrtextons = 0;
    
    if (nrhs>4) {
	    hists = (double*) mxGetData(prhs[Phistograms]); 
        nrhistograms = mxGetN(prhs[Phistograms]);
        nrtextons = mxGetM(prhs[Phistograms]);
    }
    
    fprintf(stderr,"hists %d textons %d\n",nrhistograms,nrtextons);

    if (nrGTclasses==0) nrGTclasses=1; //assume that its just an nxm array and not nxmx1
    mexPrintf("Use %d training points and\n    a pool of %d pixeldifferences (per channeltype).\n    max %d pixels in image\nmax of %d overlapping classes\n\n",(int)nrtotalpoints,nallpixdiffs,maxpixels,nrGTclasses); 


    /*** Now start training the decision tree ***/

    // initialize index holding the node where each data point ended up (so far)
	const mwSize od1[] = {1,nrtotalpoints};
	mxArray *tmp1 = mxCreateNumericArray(2, od1, mxUINT64_CLASS, mxREAL);
    uint64_t *leafindx = (uint64_t*) mxGetData(tmp1); //new unsigned long int[nrtotalpoints];

    unsigned int totalp=0;
    unsigned int totalcount=0;
    unsigned int nrtotalpos=0;
    unsigned int trainingpoints = 0;
    unsigned char maxclass=0;
    
    //fprintf(stderr,"pdectree-8\n");
    if (tps.nclasses!=0) { 
        for (unsigned int i=0; i<nrimgs; i++) {
            const int npixels = imgdimsGT[2*i]*imgdimsGT[2*i+1]; //read dimensions from GT
            unsigned char *GT = (unsigned char*) gt[i];
            for (int l=0; l<npixels; l++) {
                for (int oc=0; oc<nrGTclasses; oc++) {   //getting valid test points excluding the black region < i guess!!! >
                    if (GT[l+oc*npixels]<=tps.nclasses) {
                        nrtotalpos++;
                        break;
                    }
                }
            }
        }
    } else {
        nrtotalpos = nrtotalpoints;
    }
    
    // nrtotalpos is the number of valid points(pixels) except black pixels in all the images
    
    double *classPrior = new double[tps.nclasses]; // setting aside a memory allocation
    memset(classPrior,0,tps.nclasses*sizeof(double)); //initializing the above memory allocated to zero
    
    if ((tps.normalizeByPrior>0) && (tps.nclasses==0)) {
        mexErrMsgTxt("cannot use prior normalisation w/o specifying nr of classes\n");
    }
    
    for (unsigned int i=0; i<nrimgs; i++) {
        const int npixels = imgdimsGT[2*i]*imgdimsGT[2*i+1]; //read dimensions from GT
        unsigned char *GT = (unsigned char*) gt[i];

        for (int l=0; l<npixels; l++) {
            for (int oc=0; oc<nrGTclasses; oc++) {

                if (GT[l+oc*npixels]==0) mexErrMsgTxt("class 0 in groundtruth is not allowed");
                double u=((double) random())/RAND_MAX;
                
                if (tps.nclasses>0) {
                    //sample from positive/valid training points
                    if (GT[l+oc*npixels]<=tps.nclasses) {
                        if (tps.nrsamples>0)
			{
			  if (double(nrtotalpos-totalp)*u > (tps.nrsamples-trainingpoints))
			  {
			    leafindx[totalcount] = 0;
                          }
                          else
			  {
			    leafindx[totalcount] = 1; //default first/root node
                            trainingpoints++;
                            for (int o=0; o<nrGTclasses; o++)
			    {
			      classPrior[GT[l+o*npixels]-1]++; // calculating class priors
							// classPrior[cc] = no. of pixels belonging to that class
                             }
                          }
                            totalp++;
                        }
                        else
			  if (tps.nrsamples<0) { //use points from grid with grid distance abs(tps.nclasses)
                            int gridspace=-tps.nrsamples;
                            int x= l%imgdimsGT[2*i];
                            int y= l/imgdimsGT[2*i];
                            if ((GT[l+oc*npixels]<=tps.nclasses) && (x%gridspace==gridspace-1 && y%gridspace==gridspace-1)) {
                                leafindx[totalcount] = 1; //default first/root node
                                trainingpoints++;
                                for (int o=0; o<nrGTclasses; o++) {
                                    classPrior[GT[l+o*npixels]-1]++;
                                }
                            }
                        }
                        else
			{
                            leafindx[totalcount] = 1; //default first/root node
                            trainingpoints++;
                            for (int o=0; o<nrGTclasses; o++) {
                                classPrior[GT[l+o*npixels]-1]++;
                            }
                        }
                        break; // break out of oc, as each pixel should only be counted once, and l was counted for in this oc
                    }
                } else { //select all training data
                    if (GT[l+oc*npixels]>maxclass) maxclass = GT[l+oc*npixels];
                    leafindx[totalp] = 1; //default first/root node
                    totalp++;
                    trainingpoints++;
                    break; // break out of oc, as each pixel should only be counted once, and l was counted for this oc
                } 
            }
            totalcount++;
        }        
    }
    
    // by now leafindx is calculated
    // trainingpoints will used to train the tree (Bagging)
    
    //fprintf(stderr,"pdectree-10\n");

    if (tps.nclasses==0) tps.nclasses = maxclass; // if not already provided
    mexPrintf("Train %d classes using %d training points.\n",tps.nclasses,trainingpoints);

    // initialize data for selecting training points that need to be evaluated
    int32_t *SX = new int32_t[maxpixels]; // maxpixels = maximum no. of pixels in any image
    int32_t *SY = new int32_t[maxpixels];
    unsigned long int currentnode=1;
    
    // initialize some tree related data
    unsigned int *nodeDiffs = new unsigned int[tps.nFeatsPerNode]; // index into pool of pixel-differences

    double *resHist = new double[tps.nclasses];
    
    unsigned int threshoptimize = (tps.threshoptimize<=0) ? 2 : tps.threshoptimize; 
    // tps.threshoptimize is no. og equally spaced thresholds that is optimized over 
    // if 0 it means random thresholds
    
    uint32_t *resHistH = new uint32_t[tps.nFeatsPerNode*tps.nclasses*threshoptimize];


    // initialize array for output of feature responses
    unsigned long int treesize = (unsigned long int) pow(2,tps.maxTreeDepth+1)-1;
	const mwSize dd[] = {1,treesize};
	plhs[0] = mxCreateNumericArray(2, dd, mxINT32_CLASS, mxREAL);
    if (plhs[0]==NULL) {
        mexErrMsgTxt("Coulnt allocate output plhs[0] to store tree");
    }
	int32_t *tree = (int32_t*) mxGetData(plhs[0]);
    for (unsigned int i=0; i<dd[1]; i++) {
        tree[i] = -1; //set to unseen node
    }

    if (nlhs>=5) {
	    const mwSize dd[] = {1,treesize};
        plhs[4] = mxCreateCellArray(2, dd);
    }

    unsigned long int leafnr=0;
    plhs[1] = mxCreateCellArray(2, dd);
    if (plhs[1]==NULL) {
        mexErrMsgTxt("Coulnt allocate output plhs[1] to store leafs");
    }
    mxArray *leafNodes = plhs[1];

	plhs[2] = mxCreateNumericArray(2, dd, mxDOUBLE_CLASS, mxREAL);
    if (plhs[2]==NULL) {
        mexErrMsgTxt("Coulnt allocate output plhs[2] to store thresholds");
    }
	double *thresholds= (double*) mxGetData(plhs[2]);

//fprintf(stderr,"pdectree-11\n");
    /*** some extra output ***/

    mxArray *tmp0;
    if (nlhs>=4) {
	    tmp0 = mxCreateCellArray(2, dd);
        if (tmp0==NULL) mexErrMsgTxt("Coulnt allocate output for tree-stats");
        plhs[3] = tmp0;
    } else {
        tmp0 = NULL;
    }
    mxArray *treestats = tmp0;

	//double *X;// = new double[maxpixels*nmaxpixdiffs]; //only allocate once
	const mwSize od[] = {1,maxpixels,tps.nFeatsPerNode};
	mxArray *tmp = mxCreateNumericArray(3, od, mxDOUBLE_CLASS, mxREAL);
    if (tmp==NULL) {
        mexErrMsgTxt("Coulnt allocate output X");
    }
    double *X = (double*) mxGetData(tmp);
    if (nlhs>=6) 
        plhs[5] = tmp;

    // initialize rundom number generator
    srandom(time(NULL));
//fprintf(stderr,"pdectree-12\n");
    mexPrintf("\nNow learn the tree (maxNrNodes=%lu)...\n",treesize);
    while (currentnode<=pow(2,tps.maxTreeDepth+1)-1) {

        //check if parentnode is leafnode
        if ((currentnode>1) && isParentLeaf(leafNodes,currentnode)) {
                if (DEBUG>=50) mexPrintf("Jump node %lu since parent is leafnode\n",currentnode);
                currentnode+=2;
                continue;
        }
        //fprintf(stderr,"pdectree-13\n");

        //sample pixeldiffs for this node (Selection Sampling, see Knuth The art of computer programming Vol.2)
        // use same sampled pixeldiffs for all channeltypes, shouldnt make a difference if they are random anyway
// sub-sampling from the pool of pixel diffs:-
        unsigned int m=0;
        for (unsigned int t=0; t<nallpixdiffs; t++) {
		//fprintf(stderr,".");
            double u=((double) random())/RAND_MAX;
            if ((nallpixdiffs-t)*u > (tps.nFeatsPerNode-m)) { //I think > is required here not >= as u can equal 1
                continue;
            }
            nodeDiffs[m]=t; //select sample
            m++;
            if (m>=tps.nFeatsPerNode) {
                break; //sampling complete
            }
        }
// Randomly selecting tps.nFeatsPerNode pixel diffs for a node        
// nodeDiffs[] array containing indices of tps.nFeatsPerNode from pixDiff

        //fprintf(stderr,"pdectree-13\n");
                
        double all_bestgain=1;
        double all_entropy;
        int all_bestct = -1;
	    double *all_tt = new double[1];
        unsigned int all_bestfeat = 0;
        bool setleafIndx=false;
        double bestld=0;
        double bestrd=0;
        for (unsigned int ct=0; ct<nrchanneltypes; ct++) {
//iterate over different channeltypes and perform nodefct selection seperately then select best one

//fprintf(stderr,".");

            double bestgain;
            double entropy;
            unsigned int bestfeat = 0;

            if (DEBUG>=2) mexPrintf("\nSelect features and compute splits for node %lu\n",currentnode);
//fprintf(stderr,".");

	    memset(resHist,0,tps.nclasses*sizeof(double)); //initializing to 0
            memset(resHistH,0,tps.nclasses*tps.nFeatsPerNode*threshoptimize*sizeof(uint32_t)); // initializing to 0
            unsigned long int totalcomppixels=0;

            //iterate over images and compute feature responses
            double *threshold=NULL;
//fprintf(stderr,".");

            totalcomppixels = getFeatureResponses(X, SX, SY, resHist, resHistH, F+(ct*nallpixdiffs*pixdiffDim), 
                                                nodeDiffs, iClass[ct], images+(ct*nrimgs), 
                                                imgdims+(ct*nrimgs*2), imgdimsGT, gt, nrimgs, leafindx, tps.nFeatsPerNode, (int) tps.imageType[ct], (int) tps.imageScale[ct], currentnode, 
                                                &threshold, setleafIndx, (unsigned int) tps.nrNonZeroChannels[ct], hists, nrtextons, nrhistograms, tps, nrGTclasses);

//fprintf(stderr,".");
	    //TODO: remove only for testing
            if (nlhs>=5) {
	            const mwSize od[] = {tps.nclasses,tps.nFeatsPerNode,threshoptimize};
	            mxArray *tmp = mxCreateNumericArray(3, od, mxUINT32_CLASS, mxREAL);
                unsigned int *tmpd = (unsigned int*) mxGetData(tmp);
                memcpy(tmpd,resHistH,tps.nFeatsPerNode*tps.nclasses*threshoptimize*sizeof(uint32_t));
                mxSetCell(plhs[4],currentnode-1,tmp);
            }
//fprintf(stderr,".");
            if (totalcomppixels>1) { //NOT empty node !!!??? or node with 1 datapoint, nothing to split either
                
                //compute entropy for each pixeldiff split
                bestfeat = getSplitEntropies(resHist, resHistH, tps.nFeatsPerNode, (double) totalcomppixels, &bestgain, &entropy, &bestld, &bestrd, tps, &threshold, classPrior);
//fprintf(stderr,".");
                if (treestats!=NULL) {
	                const mwSize od[] = {1,5};
	                mxArray *tmp = mxCreateNumericArray(2, od, mxDOUBLE_CLASS, mxREAL);
                    double *tmpd = (double*) mxGetData(tmp);
                    tmpd[0] = entropy;
                    tmpd[1] = bestgain;
                    tmpd[2] = bestld;
                    tmpd[3] = bestrd;
                    tmpd[4] = (double) bestfeat;
                    mxSetCell(treestats,currentnode-1,tmp);
                }

                //mexPrintf("ct=%d; totalcomppixels=%lu; entropy=%e; bestgain=%e\n",ct,totalcomppixels,entropy,bestgain);

            }

            if (bestgain<all_bestgain) {
                all_bestgain=bestgain;
                all_entropy=entropy;
                all_bestfeat = bestfeat;
                all_bestct = ct;
                *all_tt = *(threshold+bestfeat); 
                //mexPrintf("all_tt %f \n",*all_tt);
            }

            delete[] threshold;

        } //end: for (unsigned int ct=0; i<nrchanneltypes; ct++) 
        
        //fprintf(stderr,"pdectree-14\n");

        //select best feature and compute new nodes for datapoints TODO: for now the feature responses are recomputed again
        unsigned int *pixDiff = nodeDiffs+all_bestfeat;
        memset(resHist,0,tps.nclasses*sizeof(double));
        //memset(resHistH,0,tps.nclasses*sizeof(uint32_t)*2);
        memset(resHistH,0,tps.nclasses*tps.nFeatsPerNode*2*sizeof(uint32_t));
        setleafIndx=true;
        if (DEBUG>=2) {
            mexPrintf("\n   feature for node=%lu is (%d,%d)\n",currentnode,all_bestct,*pixDiff);
        } else if (DEBUG>=1) {
            mexPrintf("."); 
            fflush(stdout);
        }
        
        //fprintf(stderr,"pdectree-15\n");
	
        tree[currentnode-1] = all_bestct*nallpixdiffs + (*pixDiff); 
        thresholds[currentnode-1] = *all_tt; 
        unsigned long int totalcomppixels = getFeatureResponses(X, SX, SY, resHist, resHistH,
                                            F+all_bestct*nallpixdiffs*pixdiffDim, pixDiff, iClass[all_bestct],
                                            images+all_bestct*nrimgs, imgdims+(all_bestct*nrimgs*2), imgdimsGT, gt, nrimgs, leafindx, 1, (int) tps.imageType[all_bestct], 
                                            (int) tps.imageScale[all_bestct], 
                                            currentnode, &all_tt, setleafIndx, (unsigned int) tps.nrNonZeroChannels[all_bestct], hists, nrtextons, 
                                            nrhistograms, tps, nrGTclasses);
        all_bestfeat = getSplitEntropies(resHist, resHistH, 1, (double) totalcomppixels, &all_bestgain, &all_entropy, &bestld, &bestrd, tps, NULL, classPrior);
        //create leaf node based on entropy gain, or if on last level, or if no split occured bestld==0 || bestrd==0
        if (DEBUG>=20) mexPrintf("bestct=%d; totalcomppixels=%lu; all_entropy=%e; all_bestgain=%e; all_tresh=%f\n",all_bestct,totalcomppixels,all_entropy,all_bestgain,*all_tt);
        if ((currentnode>pow(2,tps.maxTreeDepth)-1) || ((tps.minGain!=5) && ((totalcomppixels==0) || (all_entropy>tps.minEntropy) || (all_bestgain>=tps.minGain) || (bestld==0 || bestrd==0) ))){
	        const mwSize od[] = {1,tps.nclasses};
	        mxArray *tmp = mxCreateNumericArray(2, od, mxDOUBLE_CLASS, mxREAL);
            double *dtmp = (double*) mxGetData(tmp);
            memcpy(dtmp,resHist,tps.nclasses*sizeof(double));
            mxSetCell(leafNodes,currentnode-1,tmp);
            leafnr++;
            if (DEBUG>=1) {
                mexPrintf(" Create leafNode %lu (t=%lu;e=%e;g=%e;c=%lu;ld=%f;rd=%f)\n",leafnr,totalcomppixels,all_entropy,all_bestgain,currentnode,bestld,bestrd);
            } else {
                mexPrintf("."); 
                fflush(stdout);
            }

        }

        // change currentnode : breadth first
        currentnode++;
    }

    mexPrintf("\n\nDONE learning tree\n");

    //delete[] X;
    delete[] imgdims;
    delete[] imgdimsGT;
    delete[] images; //TODO: free the actual images as well?
    delete[] iClass;
    delete[] gt; //TODO: free the actual images as well?
    delete[] resHist;
    delete[] resHistH;
    delete[] nodeDiffs;
    delete[] SX;
    delete[] SY;
    
    fprintf(stderr,"pdectree-end\n");
}

#ifndef CMDL

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    pdDecTree(nlhs, plhs, nrhs, prhs);

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
    int nlhs=4;
    mxArray *plhs[nlhs];
    pdDecTree(nlhs, plhs, nrhs, (const mxArray**) prhs);

    for (int i=0; i<nrhs; i++) {
        mxDestroyArray(prhs[i]);
    }
    return(EXIT_SUCCESS);
}

#endif
