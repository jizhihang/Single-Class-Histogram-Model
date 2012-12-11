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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%INTEGRAL_IMAGE(I): computes the integral image for each channel of the input image I
%
%       [X] = _integral_image(I);
% 
% Returns the integral image of a multi-channel input image.      
%
%IN:
%   I - 3 dimensional image of type double, single, or uint8 with arbitrarily sized dimensions
%
%OUT:
%   X - integral image of I (same size) of type double or uint32

% $Id: integral_image.cxx,v 1.3 2008/05/08 16:29:28 schroff Exp $
*/


#include <mex.h>
#include <math.h>

#ifdef _MSC_VER
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif

template<class O, class T> static void integralImage(O *X, T *I, unsigned int sizex, unsigned int sizey, unsigned int nchannels)
{
    T *pI=I;
    O *pX=X;
    O *S=new O[sizex]; //store sum from the previous row

    for (unsigned int c=0; c<nchannels; c++) {
        O *os=S;
        for (unsigned int x=0; x<sizex; x++) {
            *(os++)=0;
        }

        for (unsigned int y=0; y<sizey; y++) {
            O colsum=0;
            os = S;
        	for (unsigned int x=0; x<sizex; x++) {
                colsum += *pI;
                *pX = colsum+*os;
                *os = *pX;
                os++; pI++; pX++;
            }
        }
    }

    delete[] S;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Check the inputs and outputs
	if ((nrhs != 1) || (mxGetNumberOfDimensions(prhs[0]) != 3))
		mexErrMsgTxt("Only one 3dim input image allowed of types double,float or uint8.");
	mxClassID class_x = mxGetClassID(prhs[0]);
	if (nrhs != 1)
		mexErrMsgTxt("Only one output argument allowed.");

    const int* dims=mxGetDimensions(prhs[0]);
	const int nchannels = dims[2];

	void *I = mxGetData(prhs[0]); //input image
	const unsigned int sizex = dims[0];
	const unsigned int sizey = dims[1];

	const int odims[] = {sizex,sizey,nchannels};
    void *X;
    
	switch (class_x) {
		case mxDOUBLE_CLASS:
	        plhs[0] = mxCreateNumericArray(3, odims, mxDOUBLE_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            integralImage((double*) X,(double*) I,sizex,sizey,nchannels);
			break;
		case mxSINGLE_CLASS:
	        plhs[0] = mxCreateNumericArray(3, odims, mxDOUBLE_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            integralImage((double*) X,(float*) I,sizex,sizey,nchannels);
			break;
		case mxUINT8_CLASS:
	        plhs[0] = mxCreateNumericArray(3, odims, mxINT32_CLASS, mxREAL);
	        X = mxGetData(plhs[0]); //output values
            integralImage((uint32_t*) X,(uint8_t*) I,sizex,sizey,nchannels);
			break;
		default:
		    mexErrMsgTxt("Only one 3dim input image allowed of types double,float or uint8.");
			break;
	}
	return;
}

