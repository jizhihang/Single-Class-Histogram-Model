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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getFeatures compute simple featureDescriptor
%		[featureDescriptor] = getFeatures(I)
%
%		- I: image to compute the featureDescriptor from
%		- opts:
%			G: Groundtruth image, for each pixel is returned
%
%This code computes featureDescriptor for a colour image I
%
% antonios filters, (Using his Lab, not the matlab ones) convolution (i.e. padding) might still be different.

function [featureDescriptor] = getFeatures_Ac17I(imagefile, varargin)

if nargin > 1
	G=varargin{1};
else
	G=inf;
end

featureDescriptor=[];

% convert to lab using a reimplementation of antonio's conversion
if ischar(imagefile)
    I=imread(imagefile);
else
    I=imagefile;
end
I=RGB2Lab(I);

F=makeAc17;

for i=1:3
	featureDescriptor(i,:,:)=imfilter(double(I(:,:,1)),F(:,:,i),'same','symmetric','conv'); %L
end
for i=1:3
	featureDescriptor(i+3,:,:)=imfilter(double(I(:,:,2)),F(:,:,i),'same','symmetric','conv');%a
end
for i=1:3
	featureDescriptor(i+6,:,:)=imfilter(double(I(:,:,3)),F(:,:,i),'same','symmetric','conv');%b channel
end

for i=4:7
	featureDescriptor(i+6,:,:)=imfilter(double(I(:,:,1)),F(:,:,i),'same','symmetric','conv');%L channel
end

featureDescriptor(14,:,:)=imfilter(double(I(:,:,1)),F(:,:,8),'same','symmetric','conv');
featureDescriptor(15,:,:)=imfilter(double(I(:,:,1)),F(:,:,9),'same','symmetric','conv');
featureDescriptor(16,:,:)=imfilter(double(I(:,:,1)),F(:,:,10),'same','symmetric','conv');
featureDescriptor(17,:,:)=imfilter(double(I(:,:,1)),F(:,:,11),'same','symmetric','conv');
