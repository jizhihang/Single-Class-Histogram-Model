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
function do_lpd(params,model);
    global CLSmapping
    mymkdir(params.model_name);

    %setting state of random no. generator algorithm in matlab
    rand('state',314159);
    
    % pixeldiff= [x1 y1 x2 y2 support1x/1y/2x/2y channel1/2; ...]
    %train trees
    disp('Train trees...')
    params.precomputedResults=1;
    forest={};
    
    if params.trainROI
        params.forestsize=params.nrObjectClasses;
    end
    starti=1;
    endi=params.forestsize;
    if params.trainNoBagging % 0 by default --> turn on bagging 1--> turn off bagging
        params.baggingNr = 0; % now no bagging
    end
    if params.baggingNr == 0 % by default 0 --> no bagging
        rrp=1:length(params.ImagesList); % rrp = list of indices of all images. train on all the images
    else % if u want bagging
% random perm of all image indices s.t. every index appears twice
        rrp=[randperm(length(params.ImagesList)) randperm(length(params.ImagesList))]; 
    end
    if params.baggingNr == 0
        params.baggingNr = length(params.ImagesList);
        bagoverlap=0;
    else % criteria to reject too small bagging number
        bagoverlap = floor((params.baggingNr*double(endi-starti)-length(params.ImagesList)+params.baggingNr)/double(endi-starti))
        if bagoverlap<0
            error('not all training images used in training, baggingNr to small or batchchunk to small');
        end
    end
    for fn = starti:endi
        fprintf('\n\nTrain tree nr: %d\n',fn);
    
        fprintf('Collect training data...\n');
        tic
        classes=[]; gt={};
        [img, params.nr_classes]=mapImageClasses('', params);% setting proper nr_classes
    
        tic
        if bagoverlap==0 % no bagging, by default
            startImg=1; endImg=length(params.ImagesList);
        else % setting images for bagging
            startImg = max(1,double((fn-starti)*(params.baggingNr-bagoverlap)+1));
            endImg   = startImg + params.baggingNr-1;
        end
        
%          totalimages = endImg - startImg + 1;
%          gt = cell(1,totalimages);
%          for i=1:totalimages
%  	  gt{i} = uint8(zeros(350,350,params.nrClasses));
%  	 end
        
        
%% computing cluster center for texton maps using k_means:-
addpath('texton');
clusters_coord = compute_clusters(params); % clusters_coord = coordinates of params.nrTextons cluster centers
model.clustercenters = clusters_coord;
save('model.mat','model');

        for img = rrp(startImg:endImg)
	fprintf('.');
            %fprintf('1\n');
%get the ground truth image of corresponding image in GT
            GT=uint8(getGT(params.ImagesList{img},params));
            if params.supervision==0 % 0 --> weak supervision only detects object classes. 1 by default
                gt{length(gt)+1}=GT; %use normal groundtruth supervision
            else
                tGT=zeros(size(GT,1),size(GT,2),params.nrClasses,'uint8')+params.nrObjectClasses+1;
                for cc = 1:params.nrClasses
                fprintf('.');
% if no of occurences of class cc in GT > 0 then set tGT denoting that class is present
                    if numel(find(GT==cc))>0 
                        tGT(:,:,cc) = cc;
                    end
                end
% gt is cell containing all such tGT images of corresponding training image
                gt{length(gt)+1}=tGT; %label all pixels to belonging to all classes in the images
            end

% till here we have computed gt s.t. dimensions of gt{i} = size(GT,1),size(GT,2),params.nrClasses
% the third dim contains info about whether a class  cc is present or not
% if gt{i}(:,:,cc) == cc ==> class cc is present in image{i}
 

% params.channels = [ textons,HOGPI-c5-n6-b4:c20-n6-b4:c40-n-12-b4 RGB AC17']
            for ft = 1:length(params.channels)
            fprintf('.');
                %fprintf('2\n');
                [featI,imageType,imageScale,nrNonZeroChannels(ft),nrchannels(ft)] = getFeatures(params.ImagesList{img},model,params,params.channels{ft});
% default output for RGB channel:
%featI = an integral image, imageType = 1, imageScale = 1, nrNonZeroChannels(ft) = 0, nrchannels(ft)=3

% intI stores the computed featI image after extracting features
                intI{length(gt),ft}=featI; % length(gt) returns the length of largest array dimesion of gt
                imageTypes(ft)=imageType;
                imageScales(ft)=imageScale;
            end
        end
% collecting featIs end here
    
        tic
        pixeldiffs=[];
        nhists=[];
        for ft = 1:length(params.channels) % params.channels = [ HOGPI-c5-n6-b4:c20-n6-b4:c40-n-12-b4 RGB AC17']
        
            %fprintf('3\n');
            %%% use patch or single-class histograms
            hists=[];
% by default these are 1000,100 and nrchannels('RGB') = 3.
% dont know why taking so less features.
            if params.ft_pertree == 0
                params.ft_pertree = nrchannels(ft);
            end
            if params.nrFeatsPerNode== 0
                params.nrFeatsPerNode = nrchannels(ft);
            end
                
            fprintf('Pixeldiffs for %s: channels=%d\n',params.channels{ft},nrchannels(ft));

            for pp=1:params.ft_pertree
            
                %fprintf('4\n');
                channels=[round(rand*(nrchannels(ft)-1))+1 round(rand*(nrchannels(ft)-1))+1]; % take two random channels
                if params.ft_pertree == nrchannels(ft)
                    channels = [pp 0];
                    if ~strcmp(params.treeHistograms,'sclass') && params.absolutePatches~=params.ft_pertree
                        error('This option is not available w/o the ''sclass'' option, or all patches absolute Patches');
                    end
                end
                for p=1:4
                
                    %fprintf('5\n');
                    rr = randperm(length(params.supportsize_eval)); %params.supportsize_eval = [10:50]
                    rp(p) = rr(1); % rp(1:4) = 4 random values from params.supportsize_eval
                end
                support1=[params.supportsize_eval(rp(1)) params.supportsize_eval(rp(2))];
                support2=[params.supportsize_eval(rp(3)) params.supportsize_eval(rp(4))];
                md = params.maxdiffdistHalf_eval; %maxdiffdistHalf_eval = [-40:40]
                for p=1:4
                    rr = randperm(length(md)); % randperm of [1:81]
                    rp(p) = rr(1); % rp(1:4) = 4 random values from [1:81]
                end
                % -floor(support/2) to correct such that rectangles are centered on "average", ie. no one sided offset
                pixeldiffs(:,pp+(ft-1)*params.ft_pertree) = [md(rp(1))-floor(support1(1)/2) md(rp(2))-floor(support1(2)/2) md(rp(3))-floor(support2(1)/2) md(rp(4))-floor(support2(2)/2) support1 support2 channels]';
            end
    
            if params.absolutePatches>0 % by default = 0
                fprintf('use %d absolute patches (out of %d) (NO Difference between rectangles)\n',params.absolutePatches,size(pixeldiffs,2));
                rp=randperm(size(pixeldiffs,2));
                pixeldiffs([7 8],rp(1:params.absolutePatches))=0; %just set support of 2nd rectangle to 0
                fprintf('    done\n');
            end
    
            pixeldiffs=int16(pixeldiffs);
        end
        tic
        
        
        
	  nclasses=params.nrClasses;
	  treeparams = {params.nrClasses,params.maxLevel,params.stopentropy,params.max_gain,params.nrFeatsPerNode,bitand(params.normalizeByPrior,1),imageTypes,imageScales,nrNonZeroChannels,params.nr_samples,params.nrThreshOpt};
	  save('images.mat','intI','gt','pixeldiffs','treeparams','nhists');
	  
	  
	  k = 400;
        nclass = 2;
        hist = zeros(k,1);
        %nhists = zeros(k,nclass);
        
        for i = 1:nclass
	  loadfile = sprintf('results/resultsforclass%d.mat',i);
	  load(loadfile);
	  x = size(result1,1);
	  y = size(result1,2);
	  hvector = reshape(result1,1,x*y);
	  for j = 1:x*y
	    hist(hvector(1,j)) = hist(hvector(1,j))+1;
	    end
	    
	  %nhists(:,i) = hist; 
	  end
	  
	 %histograms=[];
        fprintf('Now go into pdDecTree...\n');
        %fprintf('6\n');
        [tree,leaves,thresholds,treestats]=pdDecTree(intI,gt,pixeldiffs,treeparams,nhists);

        fprintf('%d pixeldiffs for %d images: %.2fs\n\n',params.ft_pertree,length(params.ImagesList),toc);
    
        tree=convertTree(pixeldiffs,nhists,tree,leaves,thresholds,treestats,params);
    
        save([params.model_name '/fn' sprintf('%03d',fn) '.mat'],'tree','treestats');
    end
    fprintf('\n');
    toc
end
