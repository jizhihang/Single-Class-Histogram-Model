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
%

function [ff]=transformForest2rfClassify(forest,params)
    for t = 1:length(forest)
        tree=forest{t};
        nNodes = 2^max(tree.T(:,1))-1;
        ff{t}.treerep=zeros(3,nNodes,'int32');
        for n = 1:size(tree.T,1)
            node = 2^(tree.T(n,1)-1)+tree.T(n,2)-1;
	        node_func=tree.T(n,3);
	        channeltype=tree.T(n,4);
            threshold = tree.T(n,5);
            ff{t}.treerep(:,node) = [channeltype node_func-1 threshold-1]; %shift node_func by 1 so that the node_funcs start with 0; leafe nodes start with -2
        end
        ff{t}.feats = tree.feats';
        ff{t}.thresholds = tree.thresholds;
        ff{t}.posteriorL=tree.L';
        ff{t}.hists = tree.hists;
        if params.ClassDistribSmoothing>0
            ss = 1./sum(ff{t}.posteriorL,1);
            ff{t}.posteriorL=ff{t}.posteriorL.*repmat(ss,size(ff{t}.posteriorL,1),1);
            ff{t}.posteriorL=ff{t}.posteriorL+params.ClassDistribSmoothing;
            ss = 1./sum(ff{t}.posteriorL,1);
            ff{t}.posteriorL=ff{t}.posteriorL.*repmat(ss,size(ff{t}.posteriorL,1),1);
        end
        if bitand(params.normalizeByPrior,2)
            tree.classPriors = tree.classPriors + params.ClassDistribSmoothing;
            tree.classPriors = tree.classPriors ./ sum(tree.classPriors);
            ff{t}.posteriorL = diag(1./tree.classPriors) * ff{t}.posteriorL; %TODO is this right?
        end
    end
end
