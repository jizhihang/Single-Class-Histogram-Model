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

%example to learn tree
ft=1; %only one channel-type here RGB
I=imread('build.jpg'); inr=1;
clear intI;
intI{inr,ft} = integral_image(I); %one image inr=1

gtROI = zeros(size(I,1),size(I,2),'uint8')+10;
gtROI(1:30,1:100) = 1; %class one: background
gtROI(165:220,200:280) = 2; %class two: fruits

%demo for overlapping classes
gtROI = cat(3,gtROI,gtROI,zeros(size(I,1),size(I,2),'uint8')+10,gtROI);
%gtROI = cat(3,gtROI,gtROI,gtROI);
GT{inr}=gtROI;

params.nrClasses=2; %0 for number of classes determined automatically
params.maxLevel=10; %max. depth of tree
params.stopentropy=0; %if entropy smaller than stopentropy in a node it is not further expanded and becomes a leaf node
params.max_gain=0; %minimum gain necessary to expand that node. otherwise turn it into leaf node
params.nrFeatsPerNode=100; %number of features selected from the pool and optimised for each node
params.normalizeByPrior=0; %if 1 then do some normalisation to avoid bias towards classes with many occurences
imageTypes(ft) = 1; %1=normal intergral image; 0 = normal image; 2= indexed image (first half index into the channel, second half gives the channel-value)
imageScales(ft) = 1; %determines the scale of the image, eg. for HOG; imageScales=5 means that one intI entry stands for 5 pixels in I
nrNonZeroChannels(ft)=0; %for imageTypes=2 it give the number of channels
params.nr_samples=0; %nr_samples=0: use all data points; nr_samples>0 sample randomly n points; nr_samples<0 sample from grid n points
params.nrThreshOpt=10; %if >0 optimise over n thresholds; if 0 select randomly; if -1 use threshold 0
nhists=[]; %used for KL-Divergence tests

treeparams = {params.nrClasses,params.maxLevel,params.stopentropy,params.max_gain,params.nrFeatsPerNode,bitand(params.normalizeByPrior,1),imageTypes,imageScales,nrNonZeroChannels,params.nr_samples,params.nrThreshOpt};

% create pool of pixeldifferences
nrchannels=size(intI,3);
params.supportsize_eval = [5:20]; %size range from which the rectangles are sampled
params.maxdiffdistHalf_eval = [-20:20]; %nr pixels each rectangle can be off-center
params.ft_pertree=20000; %features created for global pool
for pp=1:params.ft_pertree
    channels=[round(rand*(nrchannels-1))+1 round(rand*(nrchannels-1))+1]; 
    for p=1:4
        rr = randperm(length(params.supportsize_eval));
        rp(p) = rr(1);
    end
    support1=[params.supportsize_eval(rp(1)) params.supportsize_eval(rp(2))];
    support2=[params.supportsize_eval(rp(3)) params.supportsize_eval(rp(4))];
    md = params.maxdiffdistHalf_eval;
    for p=1:4
        rr = randperm(length(md));
        rp(p) = rr(1);
    end
    % -floor(support/2) to correct such that rectangles are centered on "average", ie. no one sided offset
	%pixeldiffs = [x1 y1 x2 y2 support1x support1y support2x support2y channel1 channel2; ...]
    %(x1,y1) = upper left corner of first rectangle
    %support1x support1y: extent of rectangle 
    pixeldiffs(:,pp+(ft-1)*params.ft_pertree) = [md(rp(1))-floor(support1(1)/2) md(rp(2))-floor(support1(2)/2) md(rp(3))-floor(support2(1)/2) md(rp(4))-floor(support2(2)/2) support1 support2 channels]';
end
pixeldiffs=int16(pixeldiffs);

params.verbose=0;
gt=GT;
save('/tmp/tree.mat','intI','gt','pixeldiffs','treeparams','nhists');
[tree,leaves,thresholds,treestats]=pdDecTree(intI,GT,pixeldiffs,treeparams,nhists);
f{1}=convertTree(pixeldiffs,nhists,tree,leaves,thresholds,treestats,params);
params.ClassDistribSmoothing=0.01;
forest = transformForest2rfClassify(f,params);


%for testing run:

sizex=size(I,1); sizey=size(I,2);
params.imageType=imageTypes(ft);
params.imageScale=imageScales(ft);
params.nrNonZeroChannels=nrNonZeroChannels(ft);
[classification_mex]=randomForest_mex(forest,intI,sizex,sizey,params); %classify one image
[tmp,cl]=max(classification_mex,[],1);
imshow(reshape(cl,size(I,1),size(I,2)),colorcube(params.nrClasses));
II = I;
II(:,:,1) = I(:,:,1).*uint8(1-(gtROI(:,:,1)==1));
II(:,:,2) = I(:,:,2).*uint8(1-(gtROI(:,:,1)==2));
figure, imagesc(II)

%older version
%params.precomputedResults=2;
%params.ILP=0;
%[classification,clusternr]=randomForest(f,intI,size(I,1),size(I,2),params);
