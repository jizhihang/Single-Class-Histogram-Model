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
%*/

#ifndef computeDiffs_H
#define computeDiffs_H

#ifndef CMDL
    #include <mex.h>
#endif

#include <math.h>
#include <inttypes.h>

#include "computeDiffs.h"

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

//returns the value of the sub array using integral image I
template<class T> inline double getIIvalue(T *I,const int x,const int y,const int supx,const int supy,const int sizex,const int cn) {
    const int tx1 = ((x-1));
    const int ty1 = ((y-1));
    const int tx2 = ((x+supx));
    const int ty2 = ((y+supy));

    const double d0= ((tx1<0) || (ty1<0)) ? 0 : *(I+tx1+sizex*ty1+cn);
    const double d1= (tx1<0) ? 0 : *(I+tx1+sizex*ty2+cn);
    const double d2= (ty1<0) ? 0 : *(I+tx2+sizex*ty1+cn);
    const double sum = *(I+tx2+sizex*ty2+cn) + d0 - d1 - d2;

    return sum;
}
template<class T> inline double getIIvalue(T *I,const int x,const int y,const int supx,const int supy,const int sizex,const int cn,const int imageScale) {
	//correct for parts that are not fully contained in a cellszie
	//count partials of borders
    int tx1 = (x)/(imageScale);
    int ty1 = (y)/(imageScale);
    const int tx2 = (x+supx)/(imageScale);
    const int ty2 = (y+supy)/(imageScale);

	int rx1 = imageScale-x%imageScale;
	int ry1 = imageScale-y%imageScale;
	int rx2 = (supx+1 - rx1)%(imageScale); //overlap on the other side of the rectangle
	int ry2 = (supy+1 - ry1)%(imageScale);
    if (tx1!=tx2 && rx2==0) rx2=imageScale;
    if (ty1!=ty2 && ry2==0) ry2=imageScale;
    if (rx2<0) {
        rx1=supx+1;
        rx2=0;
    }
    if (ry2<0) {
        ry1=supy+1;
        ry2=0;
    }

    double sum=0;
    bool txx=tx2-tx1-2>=0;
    bool tyy=ty2-ty1-2>=0;
    if ((supx>imageScale && supy>imageScale) && txx && tyy)
        sum = getIIvalue(I,tx1+1,ty1+1,tx2-tx1-2,ty2-ty1-2,sizex,cn) * imageScale*imageScale;
   
    sum+= getIIvalue(I,tx1,ty1,0,0,sizex,cn) * rx1*ry1; //top left corner
    sum+= getIIvalue(I,tx1,ty2,0,0,sizex,cn) * rx1*ry2; //top rightcorner
    sum+= getIIvalue(I,tx2,ty2,0,0,sizex,cn) * rx2*ry2; //bottom right corner
    sum+= getIIvalue(I,tx2,ty1,0,0,sizex,cn) * rx2*ry1; //bottom left corner

    if (tyy) {
        sum+= getIIvalue(I,tx1  ,ty1+1,0        ,ty2-ty1-2,sizex,cn) * rx1*imageScale; //top side
        sum+= getIIvalue(I,tx2  ,ty1+1,0        ,ty2-ty1-2,sizex,cn) * rx2*imageScale; //bottom side
    }
    if (txx) {
        sum+= getIIvalue(I,tx1+1,ty1  ,tx2-tx1-2,        0,sizex,cn) * ry1*imageScale; //left side
        sum+= getIIvalue(I,tx1+1,ty2  ,tx2-tx1-2,        0,sizex,cn) * ry2*imageScale; //right side
    }
    return sum;
}

//odims = output dimensions
//size(x,y) = dimensions of feature input image
template<class T, class R> void computeDiffs(R *X, int16_t *F, unsigned int *nodeDiffs, T *I, const int npixdiffs, const int sizex, const int sizey, const int32_t *sx, const int32_t *sy, const int osizex, const int osizey, const int *odims, const unsigned char imageType, const int imageScale, const unsigned int nrNonZeroCn, double *hists, unsigned int nrtextons, unsigned int nrhistograms, R *minResponse, R *maxResponse){
    
    double *tmphist=NULL;
    if (nrhistograms!=0) {
        tmphist = new double[nrtextons];
    }
    
    T *pI=I;
    //double *pX;
    for (int p=0; p<npixdiffs; p++) {
        if (minResponse != NULL) {
            minResponse[p] = 0; //std::numeric_limits<R>::max();
            maxResponse[p] = 0; //std::numeric_limits<R>::min();
        }
        unsigned int selectedDiff = p;
        if (nodeDiffs!=NULL) {
            selectedDiff = nodeDiffs[p];
        }
        int16_t *pf=F+10*selectedDiff;

        int16_t dx1 = *pf; //int16_t(floor(*pf/double(imageScale)));
        int16_t dy1 = *(pf+1); //int16_t(floor(*(pf+1)/double(imageScale)));
        int16_t dx2 = *(pf+2); //int16_t(floor(*(pf+2)/double(imageScale)));
        int16_t dy2 = *(pf+3); //int16_t(floor(*(pf+3)/double(imageScale)));
        const int16_t support1x = *(pf+4); //int16_t(floor(*(pf+4)/double(imageScale)));
        const int16_t support1y = *(pf+5); //int16_t(floor(*(pf+5)/double(imageScale)));
        const int16_t support2x = *(pf+6); //int16_t(floor(*(pf+6)/double(imageScale)));
        const int16_t support2y = *(pf+7); //int16_t(floor(*(pf+7)/double(imageScale)));
        const int16_t channel1 = *(pf+8);
        const int16_t channel2 = *(pf+9);
        const int npixels = sizex*sizey;
        const int cn1 = (channel1-1)*npixels;
        const int cn2 = (channel2-1)*npixels;

        const int sup1x = support1x-1;// >= 0 ? support1x-1 : 0;
        const int sup1y = support1y-1;// >= 0 ? support1y-1 : 0;
        const int sup2x = support2x-1;// >= 0 ? support2x-1 : 0;
        const int sup2y = support2y-1;// >= 0 ? support2y-1 : 0;
        const int nropixels = odims[0]*odims[1];

        for (int cy=0; cy<odims[1]; cy++) {
            int y = (sy!=NULL) ? *(sy+cy)-1 : cy;
            
            int y1 = y+dy1; int y2 = y+dy2;
            if ((y1<0) || (y1>osizey-support1y) || (y2<0) || (y2>osizey-support2y)) {
                if (sx!=NULL) { //otherwise do faster padding below
                    y1 = max(0,y1);
                    y1 = min(y1,osizey-support1y);
                    y2 = max(0,y2);
                    y2 = min(y2,osizey-support2y);
                } else {
                    continue; 
                }
            }
            for (int cx=0; cx<odims[0]; cx++) {
                int x = (sx!=NULL) ? *(sx+cy)-1 : cx;
                int x1 = x+dx1; int x2 = x+dx2;
                if ((x1<0) || (x1>osizex-support1x) || (x2<0) || (x2>osizex-support2x)){
                    if (sx!=NULL) { //otherwise do faster padding below
                        x1 = max(0,x1);
                        x1 = min(x1,osizex-support1x);
                        x2 = max(0,x2);
                        x2 = min(x2,osizex-support2x);
                    } else {
                        continue; //could be split into break and continue for x1<0 and x1>osizex-support (maybe faster)
                    }
        	    }
                double sum1=0;
                double sum2=0;
                if (imageType==0) { //no integral image
                    for (int c=0; c<support1y; c++) {
                        int of = sizex*((y1+c)/imageScale)+cn1;
                        for (int r=0; r<support1x; r++) {
                            pI=I+(x1+r)/imageScale+of;
                            sum1 += *pI;
                            //printf("(%d,%d)=%d\n",x1,y1,*pI);
                        }
                    }
                    for (int c=0; c<support2y; c++) {
                        int of = sizex*((y2+c)/imageScale)+cn2;
                        for (int r=0; r<support2x; r++) {
                            pI=I+(x2+r)/imageScale+of;
                            sum2 += *pI;
                            //printf("(%d,%d)=%d\n",x1,y1,*pI);
                        }
                    }
                } else if (imageType==1) {
                    //use the integral image
					if (imageScale==1) {
                        sum1 = getIIvalue(I,x1,y1,sup1x,sup1y,sizex,cn1);
                        sum2 = sup2y<0 ? 0 : getIIvalue(I,x2,y2,sup2x,sup2y,sizex,cn2);
					} else {
                        sum1 = getIIvalue(I,x1,y1,sup1x,sup1y,sizex,cn1,imageScale);
                        sum2 = sup2y<0 ? 0 : getIIvalue(I,x2,y2,sup2x,sup2y,sizex,cn2,imageScale);
					}

                    
                } else if ((imageType==2) || (imageType==3) || (imageType==4)) { //indexed channels channels 1:nnonzerochannels is index into channel nrnonzerochannels+1:2*nrnonzerochannels give the value of the channels
                    if (nrhistograms!=0) { //KL divergence between input patch/class-histograms
                        memset(tmphist,0,nrtextons*sizeof(double));
                        for (int c=0; c<support1y; c++) {
                        for (int r=0; r<support1x; r++) {
                            pI=I+(x1+r)/imageScale+sizex*((y1+c)/imageScale); //basis for this position
                            for (unsigned int nn=0; nn<nrNonZeroCn; nn++) {
                                unsigned int ch = (unsigned int) (*(pI+nn*npixels))-1;
                                tmphist[ch] += *(pI+(nn+nrNonZeroCn)*npixels);
                            }
                        }}
                        double norm=0;
                        for (unsigned int i=0; i<nrtextons; i++) {
                            norm += tmphist[i];
                        }
                        for (unsigned int i=0; i<nrtextons; i++) {
                            double b = tmphist[i]/norm;
			    fprintf(stderr,"%d ",b);
                            if (imageType==3) {
                                sum1 -= b * log((b+2.2204e-16)/(hists[(channel1-1)*nrtextons+i]+2.2204e-16));
                            } else if (imageType==4) {
                                sum1 -= (b>0) ? hists[(channel1-1)*nrtextons+i] : 0;
                            } else {
                                sum1 -= b * hists[(channel1-1)*nrtextons+i]; //compute KL(b||q1)-KL(b||q2)
                            }
                        }
                        fprintf(stderr,"\n");

                    }else{ 
                        //difference between two patches
                        for (int c=0; c<support1y; c++) {
                        for (int r=0; r<support1x; r++) {
                            pI=I+(x1+r)/imageScale+sizex*((y1+c)/imageScale); //basis for this position
                            for (unsigned int nn=0; nn<nrNonZeroCn; nn++) {
                                if (*(pI+nn*npixels)==channel1) {
                                    sum1 += *(pI+(nn+nrNonZeroCn)*npixels);
                                    break;
                                }
                            }
                        }}
                        for (int c=0; c<support2y; c++) {
                        for (int r=0; r<support2x; r++) {
                            pI=I+(x2+r)/imageScale+sizex*((y2+c)/imageScale); //basis for this position
                            for (unsigned int nn=0; nn<nrNonZeroCn; nn++) {
                                if (*(pI+nn*npixels)==channel2) {
                                    sum2 += *(pI+(nn+nrNonZeroCn)*npixels);
                                    break;
                                }
                            }
                        }}
                    }
                }
                R response=R (sum1-sum2);
                
                X[cx+odims[0]*cy+nropixels*p]=response;
                if (minResponse != NULL) {
                    minResponse[p] = (response < minResponse[p]) ? response : minResponse[p];
                    maxResponse[p] = (response > maxResponse[p]) ? response : maxResponse[p];
                }

                if (sx!=NULL) break; //break here, padding doesnt work for sx,sy specified points

                //do padding of borders
                if ((y1==0) || (y2==0)){ 
                    for (int c=cy-1; c>=0; c--)
                        X[cx+odims[0]*c+nropixels*p]=response;
                }
                if ((x1==0) || (x2==0)){
                    for (int c=cx-1; c>=0; c--) {
                        X[c+odims[0]*cy+nropixels*p]=response;
                        if ((y1==0) || (y2==0)) { 
                            for (int y=cy-1; y>=0; y--)
                                X[c+odims[0]*y+nropixels*p]=response;
                        }
                        if ((y1==osizey-support1y) || (y2==osizey-support2y)){
                            for (int y=cy+1; y<osizey; y++)
                                X[c+odims[0]*y+nropixels*p]=response;
                        }
                    }
                }
                if ((y1==osizey-support1y) || (y2==osizey-support2y)){
                    for (int c=cy+1; c<osizey; c++)
                        X[cx+odims[0]*c+nropixels*p]=response;
                }
                if ((x1==osizex-support1x) || (x2==osizex-support2x)){
                    for (int c=cx+1; c<osizex; c++) {
                        X[c+odims[0]*cy+nropixels*p]=response;
                        if ((y1==0) || (y2==0)) { 
                            for (int y=cy-1; y>=0; y--)
                                X[c+odims[0]*y+nropixels*p]=response;
                        }
                        if ((y1==osizey-support1y) || (y2==osizey-support2y)){
                            for (int y=cy+1; y<osizey; y++)
                                X[c+odims[0]*y+nropixels*p]=response;
                        }
                    }
                }

            }
        }
    }
    if (nrhistograms!=0) {
        delete[] tmphist;
    }

}
#endif
