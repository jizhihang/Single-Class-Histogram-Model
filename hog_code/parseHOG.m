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
function [parameters,imageScale,smallIScale] = parseHOG(channel,hogstr)
    if strcmp(channel,'HOG')
        parameters = [];
    else
        parameters = [];
        parameters.featureDescSave=0;
        parameters.pixelPerCell = (regexprep(hogstr,'.*c(\d+).*','$1'));
        if strcmp(parameters.pixelPerCell,hogstr)
            parameters.pixelPerCell = 8; 
        else
            parameters.pixelPerCell = str2num(parameters.pixelPerCell); 
        end

        parameters.epsilon = (regexprep(hogstr,'.*e(\d+).*','$1'));
        if strcmp(parameters.epsilon,hogstr)
            parameters.epsilon= 1750; 
        else
            parameters.epsilon= str2num(parameters.epsilon); 
        end
        
        parameters.nBins = (regexprep(hogstr,'.*n(-?\d+).*','$1'));
        if strcmp(parameters.nBins,hogstr)
            parameters.nBins= 9; 
        else
            parameters.nBins= str2num(parameters.nBins); 
        end

        parameters.nrCellsInBlock = (regexprep(hogstr,'.*b(\d+).*','$1'));
        if strcmp(parameters.nrCellsInBlock,hogstr)
            parameters.nrCellsInBlock= 4; 
        else
            parameters.nrCellsInBlock= str2num(parameters.nrCellsInBlock); 
        end

        if regexp(channel,'.*I.*')
            parameters.hogType = 'uint8';
        else
            parameters.hogType = 'double';
        end
        if regexp(channel,'.*C.*')
            parameters.grayhog = 0;
        else
            parameters.grayhog = 1;
        end
        if regexp(channel,'.*P.*') %Plain hog no resizing
            parameters.resize = 0;
            imageScale = parameters.pixelPerCell;
        else
            parameters.resize = 1;
            imageScale = 1;
        end
        smallIScale = imageScale;

        parameters.smoothMat = [];
        parameters.featureDescSave = 0;
        parameters.slideStep = parameters.pixelPerCell;
    end
end
