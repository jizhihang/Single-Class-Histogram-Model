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
treebasename = ['-ns' num2str(params.nr_samples) '-sp' num2str(params.maxdiffdistHalf)  '-fs' num2str(params.supportsize) '-ml' num2str(params.maxLevel) '-te' num2str(params.stopentropy) '-tg' num2str(params.max_gain)  '-ft' num2str(params.ft_pertree) '-fn' num2str(params.nrFeatsPerNode)  '-np' num2str(bitand(params.normalizeByPrior,1))];
if ~(length(params.channels)==1 && strcmp(params.channels{1},'RGB'))
    treebasename = [treebasename '-C'];
    for i = 1:length(params.channels)
        treebasename = [treebasename params.channels{i}];
    end
end
if ~strcmp(params.VwordAssignement,'')
    treebasename = [treebasename '-VW' params.VwordAssignement];
end
if ~strcmp(params.treeHistograms,'none')
    treebasename = [treebasename '-HH' params.treeHistograms];
end
if (params.nrThreshOpt~=0)
    treebasename = [treebasename '-TT' num2str(params.nrThreshOpt)];
end
if (params.absolutePatches~=0)
    treebasename = [treebasename '-AA' num2str(params.absolutePatches)];
end
if (params.baggingNr~=0)
    treebasename = [treebasename '-BB' num2str(params.baggingNr)];
end
if (params.trainROI~=0)
    treebasename = [treebasename '-ROI'];
end
treebasename = [treebasename textonadd];
