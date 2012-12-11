%
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
% Histogram of oriented Gradients (HOG) calculation
function hogObj = func_calcHOGDesc(img, nrCellsInBlock, pixelPerCell, slideStep, epsilon, nBins, gaussSmoothSigma, grayhog,fixedhog)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calc HOG descriptor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This funcion call bill trigg's c implementation of HoG.
% all that's done in here is to parse the long HoG descriptor back into a
% 4-dimensional matrix (x,y,cell,bin).
%
% INPUT: img, original image as RGB double
%


% ORIGINAL FUNCTION CALL FOR hog.c
% "Usage: tst_hog ...options... input-image.pgm\n"
% 	       "[-c cell-size-pixels]
%          "[-b block-size-pixels]
%          "[-s block-stride-pixels]\n"
% 	       "[-n num-orientations(-ve=>signed)]\n"
% 	       "[-d gradient-dead-zone]
%          "[-a gradient-weighting-power]\n"
% 	       "[-i spatial-interpolator(1:linear,2:sqrt,3:cubic-spline)]\n"
% 	       "[-I orientation-interpolator(1:linear,2:sqrt,3:cubic-spline)]\n"
% 	       "[-S gaussian-block-weighting-sigma]\n"
% 	       "[-k block-normalization-power]
%          "[-e block-normalization-epsilon]\n"
% 	       "[-A alpha-for-hog-display]\n");




%hog argument string
argumentList=sprintf('-b %i -c %i -s %i -k 1 -e %i -d 0 -n %i dummy.pgm', sqrt(nrCellsInBlock)*pixelPerCell, pixelPerCell, slideStep, epsilon, nBins);
argumentList = strtrim(argumentList);
nBins = abs(nBins);

if size(img,3)==1
    img = repmat(img,[1 1 3]);
end

if nargin==8 && grayhog

    %call hog with gray image
    grayImg = rgb2gray(img);
    grayImg = uint8(round(grayImg*255));
    grayImg = grayImg';
    hogVec = hogf(argumentList, grayImg); %fixed hog

else

    %call hog with rgb image
    colImg = round(img*255);
    colImg2 = permute(colImg,[3 2 1]); 
    colImg2 = uint8(colImg2);
    hogVec = hogf(argumentList, colImg2); %fixed hog

end

%reshape to 2D array
%(code to calc hogSizeX and hogSizeY as in hog.c)
nx = size(img,1);
ny = size(img,2);
xhog = sqrt(nrCellsInBlock)*pixelPerCell;
yhog = xhog;
nsteps = pixelPerCell;
xstride = slideStep;
ystride = xstride;
adx = 0;
%adx = 1;
ady = adx;
hogSizeX = floor( (nx-xhog-adx*(nsteps-1))/xstride+1 );
hogSizeY = floor( (ny-yhog-ady*(nsteps-1))/ystride+1 );   
%save desktop.mat
%error
hogVec = reshape(hogVec,nBins*nrCellsInBlock,hogSizeX*hogSizeY);
hogVec = hogVec';


%reshape vector from 2D to 4D
%slow, faster with reshape!!
hogBlockVec = zeros(hogSizeX, hogSizeY, nrCellsInBlock, nBins);
for x=1:hogSizeX
    for y=1:hogSizeY
        for nrC=1:nrCellsInBlock            
            hogInd = (x-1)*hogSizeY+y;
            cellStart = (nrC-1)*nBins+1;
            cellEnd = cellStart + nBins-1;            
            hogBlockVec(x,y,nrC,:) = hogVec(hogInd,cellStart:cellEnd);
        end        
    end
end        



%smooth along histogram bins
hogBlockVecSmoothed = [];
if gaussSmoothSigma > 0
    %calculate smoothing matrix
    if mod(max(nBins,-nBins),2) == 0
        filterLength = max(nBins,-nBins)-1; %nBins has to be odd: 3,5,7...
    else
        filterLength = max(nBins,-nBins);   
    end
    smoothVec = fspecial('gaussian',[1 filterLength], gaussSmoothSigma);
    smoothVec = [smoothVec zeros(1,max(nBins,-nBins)-length(smoothVec))];
    smoothVec = circshift(smoothVec',-floor(filterLength/2));
    smoothMat = zeros(max(nBins,-nBins));
    for i=1:max(nBins,-nBins)
        smoothMat(:,i) = circshift(smoothVec,i-1);    
    end


    %calc smoothed histogram    
    if ~isempty(smoothMat)    
        %shift and smooth the bins of the current hog template            
        hogBlockVecSmoothed = zeros(size(hogBlockVec));
        hogBlockVecPERMUTATED = permute(hogBlockVec, [4 1 2 3]);    
        for x=1:size(hogBlockVec,1)
            for y=1:size(hogBlockVec,2)
                for c=1:size(hogBlockVec,3)                
                    %slower version
                    %currHog = squeeze(hogBlockVec(x,y,c,:));                
                    %currHog = smoothMat * currHog;
                    %hogBlockVecSmoothed(x,y,c,:) = currHog;
                    currHog = hogBlockVecPERMUTATED(:,x,y,c);                
                    currHog = smoothMat * currHog;                
                    hogBlockVecSmoothed(x,y,c,:) = currHog;                
                end
            end        
        end            
    end
end


%store results in hogObj
hogObj.hogVec = hogVec;
hogObj.hogSizeX = hogSizeX;
hogObj.hogSizeY = hogSizeY;
hogObj.nrCellsInBlock = nrCellsInBlock;
hogObj.nBins = nBins;
hogObj.pixelPerCell = pixelPerCell;
hogObj.slideStep = slideStep;
%hogObj.img = img;
hogObj.hogBlockVec = hogBlockVec;
hogObj.hogBlockVecSmoothed = hogBlockVecSmoothed;


%%plot HOG descriptor
%[sumImgPlotted, hogBlockVec] = func_plotHOGDesc(img, hogVec, hogSizeX, hogSizeY, nrCellsInBlock, nBins, pixelPerCell, slideStep);
