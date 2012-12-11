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
function classification_output(resultI,I,result_name,i_base,params,GT,nameID)
    global CLSmapping;
    if nargin<7
        nameID='';
    end
    rI=reshape(resultI,size(I,1),size(I,2));

    cI = ind2rgb(rI,CLSmapping.colormap);
    si = imresize(cI,0.8,'nearest');
    imwrite(uint8(round(255*si)),[result_name i_base '@' nameID 'c.jpg']);
    imwrite(imresize(uint8(round(255*(.5.*double(I)./255+.5.*cI))),0.8,'nearest'),[result_name i_base '@' nameID 'o.jpg']);
    if nargin>5
        [nc,loc] = ismember(GT,params.classindices); 
        nc = find(loc==0);
        [ii]=find(GT~=rI);
        ii = setdiff(ii,nc);
        c=I(:,:,1); c(ii)=255-c(ii); I(:,:,1)=c;
        c=I(:,:,2); c(ii)=255-c(ii); I(:,:,2)=c;
        c=I(:,:,3); c(ii)=255-c(ii); I(:,:,3)=c;
        imwrite(imresize(uint8(I),0.8,'nearest'),[result_name i_base '@' nameID 'g.jpg']);

        R=zeros(size(GT,1),size(GT,2),3);
        c=zeros(size(GT));
        c(ii)=1; R(:,:,1)=c;
        [ii]=find(GT==rI);
        c=zeros(size(GT));
        c(ii)=1; R(:,:,2)=c;
        c(nc)=.7; R(:,:,2)=c;
        imwrite(uint8(round(255*imresize(R,0.8,'nearest'))),[result_name i_base '@' nameID 'r.jpg']);

        %figure(1),imshow(I);
        %figure(2),imshow(R);
        %figure(3),imshow(GT,CLSmapping.colormap);
    end
end
