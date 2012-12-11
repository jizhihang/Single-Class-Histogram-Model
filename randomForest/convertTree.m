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
%---------------------------------------
%

function f=convertTree(pixeldiffs,nhists,tree,leaves,thresholds,treestats,params)
    %convert to old tree format to be able to use randomForest for classification
    f.thresholds = []; th=0;
    f.L=[]; %Histogram table for the leave nodes
    f.T=[]; %[level position nodefct] storing the indices of the node evaluation functions
    f.feats=int16([]);
    f.ft=pixeldiffs';
    f.hists = nhists;
    if params.verbose>=5
        close all
        figure(333),clf;
    end
    xxx = reshape([treestats{:}],5,[])';
    entropyNorm = max(abs(xxx(:,1)));
    information_gain = max(abs(xxx(:,2)));
    fprintf('Transform tree of length %d\n',length(tree));
    modcount = round(length(tree)/100);
    
    %%% compute class priors
    priors = zeros(1,params.nrClasses);
    for l = 1:length(leaves)
        if ~isempty(leaves{l}) && ~isnan(leaves{l}(1))
            priors = priors+leaves{l};
        end
    end
    f.classPriors=priors./sum(priors);
    
    for i = 1:length(tree)
    
        if mod(i,modcount)==0
            fprintf('.');
        end
    
        level = floor(log2(i)+1);
        position = i - (2^(level-1)-1);
        child = mod(position-1,2)+1; 
        parent = floor((i-2)/2)+1;
        if parent>0 && ~isempty(leaves{parent}) %parent is leavenode skip this node
            %%mark children as filled as well for drawing purposes
            %f.leaves{2*(parent-1)+2} = nan;
            %f.leaves{2*(parent-1)+3} = nan;
            continue;
        end
        ct=0;
        if 2*(i-1)+2>length(tree) || (tree(2*(i-1)+2)==-1 && tree(2*(i-1)+3)==-1) % leave node (out of range OR no child1 AND no child2)
           f.L(size(f.L,1)+1,:) = leaves{i};
           nodefct = -size(f.L,1);
        else
            nodefct = tree(i)+1; % +1 as counted from 0 in mex-file 
            ct = floor(double(nodefct-1)/params.ft_pertree)+1;
            [tf,loc] = ismember(pixeldiffs(:,nodefct)',f.feats,'rows');
            if tf==1
                nodefct = loc;
            else
                f.feats(size(f.feats,1)+1,:) = pixeldiffs(:,nodefct);
                nodefct = size(f.feats,1);
            end
            th=th+1;
            f.thresholds(th) = thresholds(i);
        end
        f.T(size(f.T,1)+1,:) = [level position nodefct ct th];
    
        if params.verbose>=5
            % plot tree
            parameters=params;
            if nodefct<0
                parameters.level=level-1;
                parameters.position= parent - (2^(level-2)-1);
            else
                parameters.level=level;
                parameters.position = position;
            end
            parameters.child = child;
            if isempty(treestats{i}) 
                treestats{i} = zeros(1,5);
            end
            plot_treePart(333,nodefct,treestats{i}(4),treestats{i}(3),f,parameters,treestats{i}(1),treestats{1}(4)+treestats{1}(3),treestats{i}(2),entropyNorm,information_gain);
        end
    end
    if params.verbose>=5
        figure(333); print('-dpng',[params.model_name '/fn' sprintf('%03d',fn) '.png']);
    end
end
