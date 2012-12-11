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
function [hogObj, sfeatureDescriptor, label] = getFeatures_HOG(imagefile, params, varargin)

    if nargin<3 || isempty(varargin{1})
        parameters.smoothMat = [];
        parameters.epsilon = 1750;
        parameters.nBins = 6;
        parameters.nrCellsInBlock = 4;
        parameters.pixelPerCell = 8;
        parameters.slideStep = 8;
        parameters.featureDescSave = 0;
        parameters.grayhog = 1;
        parameters.resize = 1;
    else
        parameters = varargin{1};
    end
    if ~isfield(parameters,'hogType')
        parameters.hogType = 'double';
    end
    if ~isfield(parameters,'fixedhog')
        parameters.fixedhog = 0;
    end
    
    if parameters.featureDescSave>0 && (~isfield(parameters,'featureDesc_name') || isempty(parameters.featureDesc_name))
        argumentList=sprintf('-b %i -c %i -s %i -k 1 -e %i -d 0 -n %i', sqrt(parameters.nrCellsInBlock)*parameters.pixelPerCell, parameters.pixelPerCell, parameters.slideStep, parameters.epsilon, parameters.nBins);
        argumentList = regexprep(argumentList,' ','');
        parameters.featureDesc_name= ['/tmp/objrec/' params.ImagesSign '_feats_HOG__VOC/featureDescriptors/hog' argumentList '_'];
    end
    
    if nargin > 3
    	G=varargin{2};
    else
    	G=inf;
    end
    
    featureDescriptor=[];
    sfeatureDescriptor=[];
    
    if parameters.featureDescSave==1 %reload if existent
        filename=[parameters.featureDesc_name '/' filebasename(imagefile) '.mat'];
        if exist(filename,'file')
            try,
                load(filename,'featureDescriptor','sfeatureDescriptor','hogObj')
            catch,
                disp('could not load featureDescriptor, recompute them');
            end
    
        end
    end
    
    if isempty(sfeatureDescriptor)
        if ischar(imagefile)
            img = imread(imagefile);
        else    
            img = imagefile;
        end
        
        %convert image to double
        img = double(img) / 255;

        %pad image so that borders are contained in HOG
        blocksize = sqrt(parameters.nrCellsInBlock)*parameters.pixelPerCell;
        hogSizeX = ceil( (size(img,1)-blocksize)/parameters.slideStep );
        hogSizeY = ceil( (size(img,2)-blocksize)/parameters.slideStep );   
        if parameters.nrCellsInBlock>1
            hogSizeX=hogSizeX+1;
            hogSizeY=hogSizeY+1;
        end
        hogx = (hogSizeX)*parameters.slideStep+blocksize+1;
        hogy = (hogSizeY)*parameters.slideStep+blocksize+1;
        pimg = padarray(img,[hogx-size(img,1) hogy-size(img,2) 0],'symmetric','post'); %TODO this can be slightly larger then required
        
        hogObj = func_calcHOGDesc(pimg, parameters.nrCellsInBlock, parameters.pixelPerCell, parameters.slideStep, parameters.epsilon, parameters.nBins, parameters.smoothMat, parameters.grayhog,parameters.fixedhog);
        
        nrchannels = size(hogObj.hogVec,2);
        hogI = zeros(hogObj.pixelPerCell*hogObj.hogSizeX,hogObj.pixelPerCell*hogObj.hogSizeY,nrchannels,parameters.hogType);
        if strcmp(parameters.hogType,'uint8')
            hogObj.hogVec = round(hogObj.hogVec.*255);
        end

        if parameters.resize
            for x=1:hogObj.hogSizeX
            for y=1:hogObj.hogSizeY
                hogI((x-1)*hogObj.pixelPerCell+1:x*hogObj.pixelPerCell,(y-1)*hogObj.pixelPerCell+1:y*hogObj.pixelPerCell,:) = repmat(shiftdim(hogObj.hogVec(y+hogObj.hogSizeY*(x-1),:),-1),[hogObj.pixelPerCell hogObj.pixelPerCell 1]);
            end
            end
            sfeatureDescriptor = hogI(1:size(img,1),1:size(img,2),:);
        else
            sfeatureDescriptor = reshape(hogObj.hogVec,hogObj.hogSizeY,hogObj.hogSizeX,nrchannels);
            sfeatureDescriptor = permute(sfeatureDescriptor, [2 1 3]);
        end

        if parameters.featureDescSave>0 %reload or overwrite (1,2)
            mymkdir(parameters.featureDesc_name);
            save([parameters.featureDesc_name '/' getFilename(imagefile) '.mat'],'sfeatureDescriptor','featureDescriptor','hogObj')
        end
    end

end
