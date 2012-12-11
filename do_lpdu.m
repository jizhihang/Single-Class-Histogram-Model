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
function do_lpdu(params,model)
    model_name_old = params.model_name;
    params.model_name = [params.model_name '-U'];
    mymkdir(params.model_name);
    [img, params.nr_classes]=mapImageClasses('', params);

    %%% compute features in randomForest
    params.precomputedResults=2;

    starti=1;
    endi=params.forestsize;
    for fn = starti:endi
        treestats={};
        load([model_name_old '/fn' sprintf('%03d',fn) '.mat'],'tree','treestats');
        if strcmp(params.Posteriors,'reset')
            tree.L=zeros(size(tree.L,1),size(tree.L,2));
        end

        for i=1:length(params.ImagesList)
            disp(params.ImagesList{i});
            I=imread(params.ImagesList{i});
            for ft = 1:length(params.channels)
                [featI,imageType,imageScale,nrNonZeroChannels(ft),nrchannels(ft)] = getFeatures(params.ImagesList{i},model,params,params.channels{ft});
                intI{1,ft}=featI;
                imageTypes(ft)=imageType;
                imageScales(ft)=imageScale;
            end
            params.imageType = imageTypes;
            params.imageScale = imageScales;
            params.nrNonZeroChannels = nrNonZeroChannels;
            GT=getGT(params.ImagesList{i},params);
            GT=reshape(double(GT),1,[]);
            [classification,clusternr,L]=randomForest({tree},intI,size(I,1),size(I,2),params,GT);
            tree.L = L; 
        end

        save([params.model_name '/fn' sprintf('%03d',fn) '.mat'],'tree','treestats');
    end
end
