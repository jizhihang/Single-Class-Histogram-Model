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
function metarec(subfunction,varargin);

%%% Handle input parameters ***
inputParameters = {...
%shortcut, numeric, variable,               default,    description
'General options:', 0, '0', '', '',
'cf', 0, 'params.configurationFile',        '''''',     'configuration file that changes the default values',
'vb', 1, 'params.verbose',                  '0',        'verbosity',
'db', 0, 'params.Database',                 '''''',     'database ID',
'dt', 0, 'params.trainingData',             '''''',     'text-file containing list of training images',
'td', 0, 'params.testData',                 '''''',     'text-file containing list of test images',
'ns', 1, 'params.nr_samples',               '0',        'number of samples used for training',
'vw', 0, 'params.VwordAssignement',         '''''',     'type of visual word assignement (hard,soft,...)',
'sv', 1, 'params.softVariance',             '10^10',    'variance used for Gaussian soft assignement',
'ht', 0, 'params.HistogramType',            '''single''','histogram type (e.g. ''single'', ''NN'')',
'ft', 0, 'params.AppearanceFeats',          '''5x5''',   'base features used (e.g. ''3x3'',''5x5'')',
'pe', 0, 'eval',                            '''''',      'evaluate the argument string',
'bg', 1, 'params.learnBgClass',             '1',        'learn background class for VOC',
'tx', 0, 'params.textontype',         '''''',          'normal textons are k_means clustered patches',
'dp', 0, 'params.DatabasePath',             '''''',     'path to the database',
'Options for decision trees:', 0, '0'       , '', '',
'fs', 0, 'params.supportsize',              '10',       'size of support region used to compute features',
'sp', 0, 'params.maxdiffdistHalf',          '0.01',     'sparsity OR half of the maximal distance the two support regions can be away from each other',
'tg', 1, 'params.max_gain',                 '0.1',      'threshold of information gain, to stop tree growing',
'ml', 1, 'params.maxLevel',                 '20',       'maximum depth of decision tree',
'te', 1, 'params.stopentropy',              '1',        'threshold of class entropy, to stop tree growing',
'tf', 1, 'params.ft_pertree',               '10000',    'total size of pool of possible feature-functions that is created for each decision tree',
'np', 1, 'params.normalizeByPrior',         '0',        'kind of normalization that is used in the decision tree (training,testing)',
'fn', 1, 'params.nrFeatsPerNode',           '100',      'number of features that is drawn from the pool, to select the node''s decision function',
'ff', 1, 'params.forestsize',               '1',        'size of random forest that is used',
'cc', 0, 'params.channels',                 '''RGB''',  'type of channels that are used for pixel-differences (RGB,textons,...)',
'sm', 1, 'params.ClassDistribSmoothing',    '10^-10',   'if >0 smoothes the class distribution in the leave nodes',
'rr', 1, 'params.resize',                   '0',        'resize images',
'hh', 0, 'params.treeHistograms',           '''none''', 'use patch or single-class histograms',
'tt', 1, 'params.nrThreshOpt',              '0',        'number of equally spaced thresholds that is optimized over (0 for random threshold)',
'aa', 1, 'params.absolutePatches',          '0',        'number of absolute patches to use in pixelDifferences, i.e. no real pixeldifference',
'pp', 0, 'params.Posteriors',               '''reset''','reset posteriors before updating with post'
'nt', 1, 'params.nrTextons',                '8000',      'number of textons that is used',
'ts', 0, 'params.additionalForest',         '''''',     'additional forest that is used to the specified one',
'bb', 1, 'params.baggingNr',                '0',        'nr of images used for bagging in tree (to reduce memory usage); 0=no bagging',
'nb', 1, 'params.trainNoBagging',           '0',        'turn bagging off for actual training (e.g. for pmap training)',
'cm', 1, 'params.ColorModel',               '0',        'use ColorModel for mrf',
'ii', 1, 'params.integralImage',            '1',        'use integral images',
'roi', 1,'params.trainROI',                 '0',        'train ROI specific trees',
'rfc', 1,'params.rfClassify',               '1',        'use mex implementation for randomForest classification',
'ilp', 1,'params.ILP',                      '0',        'Image level prior (Shotton08), <0 power and GT-ILP, >0 proper ILP ',
'ss', 1,'params.supervision',               '0',        '1 for weak supervision, i.e. only the occurence of object-classes',
'loadC', 1,'params.loadClassification',     '0',        'loads precomputed classifications to do evaluation',
};

helpstring = [...
'metarec(''task'', [input parameter shortcut, value] pairs ...)\n'...
'  ''task'' in {''lpd'',''seg'',''eval''}'];
handle_input_parameters;
if nargin==0
    return
end
%%% END Handle input parameters ***

databaseWrapper_init;

addpath('randomForest');

%%% END Initialisation %%%

params.channels=strsplit(',',params.channels);

textonadd = '';
if ischar(params.supportsize) % we give params.supportsize = '10:50'
    params.supportsize_eval = eval(params.supportsize); %params.supportsize_eval = [10:50]
else
    params.supportsize_eval = [1:params.supportsize];
end
if ischar(params.maxdiffdistHalf) % we give params.maxdiffdistHalf = 40
    params.maxdiffdistHalf_eval=eval(params.maxdiffdistHalf);
else
    params.maxdiffdistHalf_eval=[-params.maxdiffdistHalf:params.maxdiffdistHalf]; %maxdiffdistHalf_eval = [-40:40]
end

getTreeNames;

supervisionstr='';
if params.supervision>0
    supervisionstr = ['-sprv' num2str(params.supervision) '-'];
end

params.model_name = ['/tmp/metarec/' params.Database '/modelTree-' getFilename(params.ImagesFile,'.txt') supervisionstr treebasename];
params.allResults = ['/tmp/metarec/' params.Database '/results'];
mymkdir(params.allResults);

model=[];

%params.verbose=2;

if params.verbose>0
    params
    disp(params.model_name)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The following section does the actual algorithm %%%
params.subfunction=lower(subfunction);

switch lower(subfunction)
    case {'lpdu'} %just update the posteriors
        if params.supervision>0
            error('unsupervised not possible for lpdu');
        end
        do_lpdu(params,model);

    case {'lpd'}
        do_lpd(params,model);

    case {'treestats'}
        do_treestats(params,model);
        
	case {'eval','evalu'}
        do_eval(params,model,subfunction);

	otherwise
		disp('Requested method not known!')
        return
end

disp('    done!');
