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
function do_eval(params,model,subfunction)
    global CLSmapping;
    addpath('randomForest');
    %stats(1) = evalseg('init',params,'',[params.Database]);
    [resim,nrclasses] = mapImageClasses('',params);
    stats.confcounts = zeros(nrclasses);
    forest={};
    forest2={};
    origsupportsize = params.supportsize_eval;
    origmaxdiffDistHalf = params.maxdiffdistHalf;

    if strcmp(lower(subfunction),'evalu')
        params.model_name = [params.model_name '-U']
    end

    for fn = 1:params.forestsize
        if strcmp(params.treeHistograms,'sclassE') && ~strcmp(lower(subfunction),'evalu')
            params.supportsize_eval = origsupportsize(fn);
            params.maxdiffdistHalf = -floor(origsupportsize(fn)/2);
            tree = makeKLTree(model,params);
        else
            fname = [params.model_name '/fn' sprintf('%03d',fn) '.mat'];
            if ~exist(fname,'file')
                fname
                warning('Not all trees could be loaded!');
                continue;
            else
                fprintf('Load tree %d\n',fn);
                load(fname,'tree');
                if isfield(tree,'tree')
                    tree = rmfield(tree,'tree');
                    tree = rmfield(tree,'leaves');
                end
            end
        end
        if (size(tree.T,2)<4)
            tree.T = [tree.T ones(size(tree.T,1),1)];
        end
        forest{length(forest)+1}=tree;

        if ~strcmp(params.additionalForest,'')
            fname = [vgg_dirname(params.model_name) '/' params.additionalForest '/fn' sprintf('%03d',fn) '.mat'];
            load(fname,'tree');
            forest2{length(forest2)+1}=tree;
        end
    end

    if params.rfClassify
        forest = transformForest2rfClassify(forest,params);
    end
    fprintf('Use %d trees\n',length(forest));

    result_name = [params.model_name '/' regexprep(params.testData,'/','-') '-ff' num2str(length(forest)) '-np' num2str(params.normalizeByPrior) '-sm' num2str(params.ClassDistribSmoothing)];
    if ~isempty(forest2)
        result_name = [result_name '-ForestComb/'];
    end
    result_name = [result_name '/'];
    mymkdir(result_name);

    for fn = 1:length(forest)
        if isempty(forest{fn}.feats)
            warning(['Tree' num2str(fn) 'only has one node.']);
        end
        if ~isfield(forest{fn},'hists')
            forest{fn}.hists=[];
        end
    end

    if params.ILP>0
        all_ILPs = get_ILP(params);
    end


    % classify all images with each tree
    starti=1;
    endi=length(params.ImagesListTest);
    for img = starti:endi 
    	base_imagename=filebasename(params.ImagesListTest{img});
    	I=imread(params.ImagesListTest{img});
    	fprintf('%s\n',base_imagename);
        intI={};
        %% computing cluster center for texton maps using k_means:-
addpath('texton');
load('model.mat');
        
        
        for ft = 1:length(params.channels)
            [featI,imageType,imageScale,nrNonZeroChannels(ft),nrchannels(ft)] = getFeatures(params.ImagesListTest{img},model,params,params.channels{ft});
            intI{1,ft}=featI;
            imageTypes(ft)=imageType;
            imageScales(ft)=imageScale;
        end

        %%% compute features in randomForest
        params.precomputedResults=2;
        params.imageType = imageTypes;
        params.imageScale = imageScales;
        params.nrNonZeroChannels = nrNonZeroChannels;

        if params.loadClassification
            load([result_name '/' base_imagename '.mat'],'classification');
        else

            fprintf('Classify: ');
            tic
            if params.rfClassify
                sizex=size(I,1); sizey=size(I,2);
                %save('~/temp/randomForest_mex.mat','forest','intI','sizex','sizey','params');
                [classification]=randomForest_mex(forest,intI,sizex,sizey,params);
            else
                [classification,clusternr]=randomForest(forest,intI,size(I,1),size(I,2),params);
            end
            toc
            if ~isempty(forest2)
                [classification2,clusternr]=randomForest(forest2,intI,size(I,1),size(I,2),params);
                combinedclasses = classification + classification2;
                classification = combinedclasses.*repmat(1./sum(combinedclasses,1),size(combinedclasses,1),1);

            end
            
            classification = classification(params.classindices,:);
            normer = sum(classification,1);
            for c=1:size(classification,1)
                classification(c,:) = classification(c,:)./normer;
            end
        end

        resultI=getMap(classification',params);

        resultI=mapImageClasses(resultI,params);
        i_base = base_imagename;
        I=imread(params.ImagesListTest{img});
        if params.verbose>=2
            GGT=getGT(params.ImagesListTest{img},params);
            classification_output(resultI,I,result_name,i_base,params,GGT);
        end
        if params.verbose>=3
            if ~params.loadClassification
                save([result_name '/' base_imagename '.mat'],'classification','params');
            end
        end
        if params.verbose>=5
            figure(555), subplot(1,2,1), imshow(reshape(resultI,size(I,1),size(I,2)),CLSmapping.colormap);
            subplot(1,2,2), imshow(I);
            drawnow;
        end
        if params.verbose>=10
            figure(133); print('-dpng',[result_name i_base '-fn' num2str(fn) '.png']);
        end
        %stats(1) = evalseg('eval',params,stats(1),params.ImagesListTest{img},resultI);
        [gtim,tmp,locs] = getGT(params.ImagesListTest{img},params);
        resim = reshape(resultI,size(gtim,1),size(gtim,2));
        sumim = double(gtim)+(double(resim)-1)*nrclasses; %don't need +1 here since bg-class is at the end; instead (resim-1) to count first class at column 1
        hs = histc(sumim(locs),1:nrclasses*nrclasses); 
        stats.confcounts(:) = stats.confcounts(:) + hs(:);

    end

    %stats(1) = evalseg('finish',params,stats(1));
    labels={};
    for i = 1:length(params.classindices) %-1 to remove background class
        [tmp,tmp2,tmp3,labels{i}]=classmapping(i,params);
    end
    confcounts_=stats.confcounts(params.classindices,params.classindices); 
    if sum(stats.confcounts(:))~=sum(confcounts_(:))
        error('OhOHHHHHH, classified some classes that arent in params.classindices');
    end
    stats.confcounts=confcounts_;
    anum=size(stats.confcounts,1);
    
    fid = 1; %stdout
    
    fprintf(fid,'----------------------------------------------\n');
    fprintf(fid,'Evaluate: \n');
    % confusion matrix - first index is true label, second is inferred label
    conf = zeros(anum);
    rawcounts = stats.confcounts;
    stats.overall_acc = 100*sum(diag(stats.confcounts)) / sum(stats.confcounts(:));
    fprintf(fid,'Percentage of pixels correctly labelled overall: %6.3f%%\n',stats.overall_acc);
    stats.accuracies = zeros(length(params.classes),1);
    fprintf(fid,'Percentage of pixels correctly labelled for each class\n');
    for j=1:anum
       rowsum = sum(stats.confcounts(j,:)); 
       if (rowsum>0), conf(j,:) = 100*stats.confcounts(j,:)/rowsum;   end;
       stats.accuracies(j) = conf(j,j);
       %clname = 'background';
       %if (j>1), clname = params.classes{j-1};end;
       [tmp,tmp2,tmp3,clname]=classmapping(j,params);
       fprintf(fid,'  %14s: %6.3f%%\n',clname,stats.accuracies(j));
    end
    stats.accuracies = stats.accuracies(1:end);
    stats.avacc = mean(stats.accuracies);
    fprintf(fid,'-------------------------\n');
    fprintf(fid,'Average accuracy: %6.3f%%\n',stats.avacc);
end
