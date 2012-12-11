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
clear
clc

nrtrees=10;
nrsamples=10000;
maxdepth=15;
featpertree=30000;
featpernode=200;
nrtextons = 300;

features = 'textons,HOGPI-c5-n6-b4:c20-n6-b4:c40-n-12-b4,RGB,AC17';
%this code should run out of the box for the following database
%http://research.microsoft.com/en-us/um/people/antcrim/data_objrec/msrc_objcategimagedatabase_v2.zip
databasepath = '~/databases/'; %please adjust accordingly

verbose = 2;

tic;

metarec('lpd','cc',features,'db','ms_db','dt','UID_train','td','UID_test','fs','10:50','sp',40,'ml',maxdepth,'tf',featpertree,'fn',featpernode,'ff',nrtrees,'tg',0,'te',0,'vb',verbose,'np',0,'ns',nrsamples,'sm',0.1,'tt',100,'dp',databasepath,'nt',nrtextons)
fprintf('First part training done: '); toc

metarec('eval','cc',features,'db','ms_db','dt','UID_train','td','UID_test','fs','10:50','sp',40,'ml',maxdepth,'tf',featpertree,'fn',featpernode,'ff',nrtrees,'tg',0,'te',0,'vb',verbose,'np',0,'ns',nrsamples,'sm',0.1,'tt',100,'dp',databasepath,'nt',nrtextons)
fprintf('Finished classification: ');

%this should give a performance around 60% (pixels correctly overall)
%and takes around 6h (training) on a current desktop

%update the posteriors in the leaf nodes using all training pixels
%metarec('lpdu','cc',features,'db','ms_db','dt','UID_train','td','UID_test','fs','10:50','sp',40,'ml',maxdepth,'tf',featpertree,'fn',featpernode,'ff',nrtrees,'tg',0,'te',0,'vb',verbose,'np',0,'ns',nrsamples,'sm',0.1,'tt',100,'dp',databasepath,'nt',nrtextons)
%fprintf('Updated leaf posteriors done: '); toc

%evaluate using updated decision trees
%metarec('evalu','cc',features,'db','ms_db','dt','UID_train','td','UID_test','fs','10:50','sp',40,'ml',maxdepth,'tf',featpertree,'fn',featpernode,'ff',nrtrees,'tg',0,'te',0,'vb',verbose,'np',0,'ns',nrsamples,'sm',0.1,'tt',100,'dp',databasepath,'nt',nrtextons)
%fprintf('Finished classification: ');

