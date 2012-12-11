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
%reimplementation of the RGB2Lab that was used in Winn05 (Antonio's ICCV05 paper)

function LabI=RGB2Lab(I)
	xyzI=RGBtoXYZ(double(I)./255.0);
	LabI=XYZtoLab(xyzI);
end

function xyzI=RGBtoXYZ(I)
	%GAMMA=false;
	%if(GAMMA), GammaCorrect(R,G,B,2.2); end;

	xyzI(:,:,1) =  0.412453 * I(:,:,1) + 0.357580 * I(:,:,2) + 0.180423 * I(:,:,3); %x
	xyzI(:,:,2) =  0.212671 * I(:,:,1) + 0.715160 * I(:,:,2) + 0.072169 * I(:,:,3); %y
	xyzI(:,:,3) =  0.019334 * I(:,:,1) + 0.119193 * I(:,:,2) + 0.950227 * I(:,:,3); %z
end

function LabI=XYZtoLab(I)
	Xn = 0.950456; Yn = 1.000; Zn = 1.088854;
	Thr=0.008856; r13=1.0/3.0; r16116=16.0/116.0;
	X=I(:,:,1); Y=I(:,:,2); Z=I(:,:,3);
	
	xr = X./Xn; yr = Y./Yn; zr = Z./Zn; % ratios wrt reference white
	
	% computing Lab
	yr_Thr=yr>Thr;
	L = double(yr_Thr).*(116.0 * yr.^r13 - 16.0) + double(~yr_Thr).*(903.3 * yr);

	xr_Thr=xr>Thr;
	fxr = double(xr_Thr).*(xr.^r13) + double(~xr_Thr).*(7.787*xr + r16116);

	yr_Thr=yr>Thr;
	fyr = double(yr_Thr).*(yr.^r13) + double(~yr_Thr).*(7.787*yr + r16116);

	zr_Thr=zr>Thr;
	fzr = double(zr_Thr).*(zr.^r13) + double(~zr_Thr).*(7.787*zr + r16116);

	a = 500.0 * (  fxr - fyr );
	b = 200.0 * (  fyr - fzr );

	LabI(:,:,1)=L;
	LabI(:,:,2)=a;
	LabI(:,:,3)=b;
end

function gI=GammaCorrect(I, gamma)
	gI=I.^gamma;
end
