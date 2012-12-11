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
if nargin==0
    disp('--------------------------------------------------------------------------------');
    disp('Help and default parameters');
    fprintf('%s\n',helpstring);
    disp('Input Parameters:');
    ip = inputParameters';
    for i = 1:size(inputParameters,1)
        fprintf('%s\t%s [%s]\n',ip{1,i},ip{5,i},num2str(eval(ip{3,i})));
    end
    return;
end

if nargin > 1
    clear opts;
    %%% first get configuration file
    for i = 1:length(varargin)
        if strcmp(varargin{i},'cf')
            params.configurationFile = varargin{i+1};
            eval(params.configurationFile);
            break;  
        end
    end

    ip={};
    opts=struct(varargin{:});
    for i = 1:size(inputParameters,1)
        if strcmp(inputParameters{i,3},'0'), continue, end;

        %%% create variables and assign default, if not existent already
        if ~exist(inputParameters{i,3},'var')
            eval([inputParameters{i,3} ' = ' inputParameters{i,4} ';']);
        end

        if ~isfield(opts,inputParameters{i,1})
            opts = setfield(opts,inputParameters{i,1},eval(inputParameters{i,3})); 
        end
    end
    opts=inputParameters_typeconversion(inputParameters,opts);
    for i = 1:size(inputParameters,1)
        if strcmp(inputParameters{i,3},'0'), continue, end;
        if strcmp(inputParameters{i,3},'eval'), 
            if isfield(opts,'pe')
                eval(opts.pe);
            end
            continue
        end;
        eval([inputParameters{i,3} ' = opts.' inputParameters{i,1} ';'])
    end
end
