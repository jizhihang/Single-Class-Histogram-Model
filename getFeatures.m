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
function [featI,imageType,imageScale,nrNonZeroChannels,nrchannels,I] = getFeatures(image,model,params,channel)
    global FLOS_textonsSave;
    try,
    I=imread(image);
     catch
        try,
            I = im2double(imread(regexprep(image,'\.gif.*$','.png')));
        catch
            disp(['could not load: ' image]);
            system(['convert ' image ' ' regexprep(image,'\.jpg$','.png')]);
            I = im2double(imread(regexprep(image,'\.jpg$','.png')));
        end
    end
    if params.resize % by default it is 0
        if length(params.resize)==1
            xborder = round(max(0,size(I,2)-size(I,1))/2);
            yborder = round(max(0,size(I,1)-size(I,2))/2);
            pimg = double(padarray(I,[xborder yborder 0],'symmetric','both')); %make it square
            xborder=xborder*params.resize/size(pimg,1);
            yborder=yborder*params.resize/size(pimg,2);
            I=imresize(double(pimg)/255,[params.resize, params.resize]);
        else
            I=imresize(double(I)/255,params.resize);
        end
        warning off;
        I=uint8(round(I*255));
        warning on;
    end
    nrNonZeroChannels = 0; % o/w changes only in textons
    imageScale = 1; % o/w changes only in HOG

%integral image is a representation for quickly and efficiently generating the sum of values in a rectangular subset of a grid
%the value at any point (x, y) in the integralImage(summed area table) is just the sum of all the pixels above and to the left of (x, y), inclusive
%Integral Image I can be computed in a single pass over input image 'i' using the recursion:-
% I(x,y) = i(x,y) + I(x-1,y) + I(x,y-1) - I(x-1,y-1)


%params.integralImage is by default 1 => use Integral Image

    if strcmp(channel,'pmap')
        imageType = 1; %normal intergral image
        if regexp(params.subfunction,'eval','ONCE')
            data=params.testData;
        else
            data=params.trainingData;
        end
        result_name = [regexprep(params.model_name,'pmap','') '/' regexprep(data,'/','-') '-ff' num2str(params.forestsize) '-np' num2str(params.normalizeByPrior) '-sm' num2str(params.ClassDistribSmoothing)];
        load([result_name '/' filebasename(image) '.mat'],'classification');
        featI=shiftdim(reshape(classification,[],size(I,1),size(I,2)),1);
        featI=integral_image(featI);
        nrchannels = size(featI,3);
    elseif strcmp(channel,'textons')
        clear II;
        FLOS_textonsSave = 1;
        nrchannels=size(model.clustercenters,2);
        addpath('texton');
        itx=computeTextonMap(image,model.clustercenters,1,params);
        imageType = 2; %indexed image %first half indeces the channel, second half gives the channel-value
        if regexp(params.VwordAssignement,'soft','ONCE')
            nrNonZeroChannels = double(size(itx,3)/2);
            featI=itx;
            %featI=int32(itx);
        else
            II=cat(3,itx,ones(size(itx,1),size(itx,2)));
            nrNonZeroChannels=1;
            featI=int32(II);
        end
    elseif strcmp(channel,'AC17')
        fI = getFeatures_Ac17I(image);
        if params.integralImage
            featI=integral_image(shiftdim(fI,1));
            imageType = 1; %normal intergral image
        else
            featI=shiftdim(fI,1);
            imageType = 0;
        end
        nrchannels = size(featI,3);
    elseif strcmp(channel,'LAB')
        Ilab = RGB2Lab(I);
        if params.integralImage
            featI=integral_image(Ilab);
            imageType = 1; %normal intergral image
        else
            featI=Ilab;
            imageType = 0;
        end
        nrchannels = size(featI,3);

% for RGB channel:
% returns Integral image by default o/w returns original image
    elseif strcmp(channel,'RGB') 
        if params.integralImage
            featI=integral_image(I);
            imageType = 1; %normal intergral image
        else
            featI=I;
            imageType = 0;
        end
        nrchannels = size(featI,3);
    elseif regexp(channel,'HOG','ONCE')
        warning off;
        addpath('hog_code');
        warning on;
        hogs=strsplit(':',channel);
        featI = [];
        
        for hi = 1:length(hogs)
            [parameters,imageScale,smallIScale] = parseHOG(channel,hogs{hi});
            parameters.featureDescSave=1;
            if params.resize
                parameters.featureDescSave=0;
                [hogObj,shogI] = getFeatures_HOG(I,params,parameters);
            else
                [hogObj,shogI] = getFeatures_HOG(image,params,parameters);
            end
            if length(hogs)>1 && ~isempty(featI) && ~isempty(regexp(channel,'.*P.*'))
                %need to resize "coarser" hogs for HOGPI
                smallIScale = (regexprep(hogs{1},'.*c(\d+).*','$1'));
                if strcmp(smallIScale,hogs{1}), error('HOG needs to be fully specified'); end
                smallIScale = str2num(smallIScale);
                if mod(imageScale,smallIScale)~=0, error('bigger hogs need to be dividable by smaller hogs'); end
                mult = uint8(imageScale/smallIScale);
                intI=[];
                for x=1:size(shogI,1)
                for y=1:size(shogI,2)
                    intI((x-1)*mult+[1:mult],(y-1)*mult+[1:mult],:) = repmat(shogI(x,y,:),[mult,mult,1]);
                end
                end
                shogI = intI(1:size(featI,1),1:size(featI,2),:);
            end
            if params.integralImage
                intI=integral_image(shogI);
                imageType = 1; %normal intergral image
            else
                intI=shogI;
                imageType = 0;
            end
            featI=cat(3,featI,intI);
        end
        imageScale = smallIScale;
        nrchannels = size(featI,3);
    end
    if params.resize
        if length(params.resize)==1
            %set border HOGs to zero
            featI(1:ceil(xborder/parameters.pixelPerCell),:,:)=0;
            featI(end-ceil(xborder/parameters.pixelPerCell)+1:end,:,:)=0;
            featI(:,1:ceil(yborder/parameters.pixelPerCell),:)=0;
            featI(:,end-ceil(yborder/parameters.pixelPerCell)+1:end,:)=0;
        end
    end
end
