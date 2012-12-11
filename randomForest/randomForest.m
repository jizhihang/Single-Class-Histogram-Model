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
%function [classes]=randomForest(forest,D)
%   forest= cellarray of trees
%   D = data (each column one datum)

function [combinedclasses,clusternr,L]=randomForest(forest,D,sizey,sizex,params,GT)
    if ~isfield(params,'precomputedResults');
        params.precomputedResults=0;
    end
    if params.precomputedResults==2
        nrPoints = sizex*sizey;
    else
        nrPoints = size(D{1},2);
    end
    if nargin>5 %instead of classification do update of the posteriors
        if nrPoints~=length(GT)
            error('number of training data and GT need to match');
        end
        if params.nrClasses ~= size(forest{1}.L,2)
            error('number of classes to learn does not match');
        end
    else
        GT=[];
    end
    combinedclasses=zeros(size(forest{1}.L,2),nrPoints);
    allfuncs={};
    for t = 1:length(forest)
        tree=forest{t};
        allfuncs{t}=[];
	    params.level=1;
	    params.position=1;
	    params.child=1;
        splitindices=1:nrPoints;
        leftnodedata=[];rightnodedata=[];
        stack={};stackpos=1; %stack for non recursive version
        stack{stackpos}.data=splitindices;
        stack{stackpos}.level=1; stack{stackpos}.child=1; stack{stackpos}.position=1;
        if params.verbose>=10
            figure(133), clf(133);
        end
        oldparent=0;
        maxlevel=max(tree.T(:,1));
        classes=zeros(size(forest{1}.L,2),nrPoints);
        while 1 %iterative version of tree growing
            %pop from stack
            splitindices=stack{stackpos}.data;
            params.child=stack{stackpos}.child; params.level=stack{stackpos}.level; params.position=stack{stackpos}.position;
        	[cl,node_func]=evaluateLevel; 
            allfuncs{t}(length(allfuncs{t})+1)=node_func;
            if ~isempty(cl)
                classes(:,splitindices)=repmat(cl,1,length(splitindices)); %actuall classification
                clusternr(splitindices)=-node_func;
            end
            stackpos=stackpos-1;
            %push on stack
            if ~isempty(rightnodedata)
                stackpos=stackpos+1;
                stack{stackpos}.data=rightnodedata;
                stack{stackpos}.level=params.level; stack{stackpos}.position=params.position;
                stack{stackpos}.child=2;
            end
            if ~isempty(leftnodedata)
                stackpos=stackpos+1;
                stack{stackpos}.data=leftnodedata;
                stack{stackpos}.level=params.level; stack{stackpos}.position=params.position;
                stack{stackpos}.child=1;
            end
            
            if params.verbose>=10
                plot_treePart(133,node_func,length(rightnodedata),length(leftnodedata),tree,params,0,nrPoints)
                if params.verbose>=15
                    pause
                end
            end

            if stackpos==0 %stack is empty
                break;
            end
        end
        if params.ClassDistribSmoothing>0
            classes=classes.*repmat(1./sum(classes,1),size(classes,1),1);
            classes=classes+params.ClassDistribSmoothing;
            classes=classes.*repmat(1./sum(classes,1),size(classes,1),1);
        end
        if bitand(params.normalizeByPrior,2)
            tree.classPriors = tree.classPriors + params.ClassDistribSmoothing;
            tree.classPriors = tree.classPriors ./ sum(tree.classPriors);
            classes = classes' * diag(1./tree.classPriors);
            classes = classes';
        end
        if params.ILP~=0
            classes = classes' * diag(params.ILPDistribution.^abs(params.ILP));
            classes = classes';
        end
        combinedclasses=combinedclasses+classes;
        forest{t} = tree;
    end
    combinedclasses=combinedclasses.*repmat(1./sum(combinedclasses,1),size(combinedclasses,1),1);
    L=tree.L;

function [allclasses,node_func]=evaluateLevel
    allclasses=[];
    leftnodedata=[];rightnodedata=[];
    l=find(tree.T(:,1)==params.level); %find level
    p=find(tree.T(l,2)==nodeindex(params.position,params.child)); %find position
    if isempty(p)
        warning('mähhh unseen node');
		allclasses=ones(size(tree.L,2),1)*1/size(tree.L,2); %uniform distribution
        node_func=0;
        return
    end
	node_func=tree.T(l(p),3);
	channeltype=tree.T(l(p),4);
    if size(tree.T,2)<5 %for backwards compatibility
        threshold=node_func;
    else
	    threshold=tree.T(l(p),5);
    end
	if node_func<0 %leave node
        if ~isempty(GT) %instead of classification do update of the posteriors
            ss = find(GT(splitindices)<=params.nrClasses); %only use objectClasses and possibly background
            if ~isempty(ss)
                %update class posteriors
                dd = GT(splitindices(ss));
                einsen=ones(size(dd));
                h=full(sparse(einsen,double(dd),einsen,1,params.nrClasses));
                tree.L(-node_func,:) = tree.L(-node_func,:)+h;
            end

		    allclasses=ones(size(tree.L,2),1)*1/size(tree.L,2); %uniform distribution
        else
		    c=tree.L(-node_func,:);
		    allclasses=c';
            if sum(allclasses)==0
                allclasses = allclasses + 1/length(allclasses);
                warning('empty leave node');
            end
        end
		return
	end
    if params.precomputedResults==0
	    result=tree.ft(node_func,:)*[D{t}(:,splitindices); ones(1,length(splitindices))]; %evaluate all possible features with one matrix multiplicaion
    elseif params.precomputedResults==1
        result=D{t}(node_func,splitindices);
    elseif params.precomputedResults==2
        [sx,sy]=ind2sub([sizey sizex],splitindices);
        sx=int32(sx);
        sy=int32(sy);
        if channeltype == 0
            keyboard
        end
        result = precomputeResultsFromPixelDiff(double(D{1,channeltype}),sizey,sizex,int16(tree.feats(node_func,:))',params.imageType(channeltype),params.imageScale(channeltype),params.nrNonZeroChannels(channeltype),tree.hists,sx,sy);
    end
	leftnodedata=splitindices(find(result<tree.thresholds(threshold)));
	rightnodedata=splitindices(find(result>=tree.thresholds(threshold)));
    
    params.level=params.level+1; %next level
	params.position=nodeindex(params.position,params.child); %corresponding offset in next level
end

end
