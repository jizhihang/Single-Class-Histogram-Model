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
% based on makeRFSfilters from Varma
% implements the 17 dimensinal filterbank used in Winn, Criminisi ICCV 2005
%
% To convolve an image I with the filter bank you can either use the matlab function
% conv2, i.e. responses(:,:,i)=conv2(I,F(:,:,i),'valid'), or use the Fourier transform.
function F=makeAc17

  SUP=49;                 % SUPport of the largest filter (must be odd)
  ELONGATION=3;		% for gaussian derivatives
  SCALE_gauss=[1,2,4];   % Sigma_{x} for the gauss kernels
  SCALE_log=[1,2,4,8];   % Sigma_{x} for the log
  SCALE_gd=[2,4];        % Sigma_{x} for the gaussian derivatives

  F=zeros(SUP,SUP,11);
  hSUP=(SUP-1)/2;
  [x,y]=meshgrid([-hSUP:hSUP],[-hSUP:hSUP]);
  orgpts=[x(:) y(:)];

  count=1;
  for scale=1:length(SCALE_gauss),
  	F(:,:,count)=(fspecial('gaussian',SUP,SCALE_gauss(scale)));
    count=count+1;
  end
  for scale=1:length(SCALE_log),
  	F(:,:,count)=(fspecial('log',SUP,SCALE_log(scale)));
    count=count+1;
  end
  for scale=1:length(SCALE_gd),
	gdy=acgaussderkernel_y(SUP,ELONGATION*SCALE_gd(scale),SCALE_gd(scale));
	gdx=acgaussderkernel_x(SUP,SCALE_gd(scale),ELONGATION*SCALE_gd(scale));
 	F(:,:,count)=gdx;
	count=count+1;
 	F(:,:,count)=gdy;
	count=count+1;
  end
return

