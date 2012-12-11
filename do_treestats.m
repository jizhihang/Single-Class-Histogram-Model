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
function do_treestats(params,model)
    global CLSmapping;
    resultsfile = [params.model_name '/foreststats.txt'];
    if ~exist(params.model_name,'dir')
        error('model doesnt exist');
    end
    fid = fopen(resultsfile,'w');
    histcomb={};
    channelcounts=[];
    for fn = 1:params.forestsize
        fprintf(fid,'%s\n',['Show tree nr: ' num2str(fn)]);
        treestats={};
        load([params.model_name '/fn' sprintf('%03d',fn) '.mat'],'tree','treestats');

        h=hist(tree.T(:,4),1:length(params.channels));
        for cc = 1:length(params.channels)
            for level = 1:max(tree.T(:,1)-1)
                li = find(tree.T(:,1)==level);
                hh=hist(tree.T( li ,4 ),1:length(params.channels));
                channelcounts(cc,fn,level) = hh(cc);

                node_fct = tree.T( li, 3 );
                node_fct = node_fct(find(node_fct>0));
                fts = double(tree.ft(node_fct,:));
                %correct the offsets to be more intuitive
                recpos = double([(fts(:,1)+floor(fts(:,5)./2)) (fts(:,2)+floor(fts(:,6)./2)) (fts(:,3)+floor(fts(:,7)./2)) (fts(:,4)+floor(fts(:,8)./2))]);
                fts = double([abs(recpos(:,1)) abs(recpos(:,2)) abs(recpos(:,3)) abs(recpos(:,4)) sqrt((recpos(:,1)-recpos(:,3)).^2+(recpos(:,2)-recpos(:,4)).^2) fts(:,5) fts(:,6) min(fts(:,5),fts(:,6))./max(fts(:,5),fts(:,6)) fts(:,7) fts(:,8) min(fts(:,7),fts(:,8))./max(fts(:,7),fts(:,8))]);
                mean_fts(cc,fn,level,:)=mean(fts,1);
                std_fts(cc,fn,level,:)=std(fts,0,1);

            end
            fprintf(fid,'%40s: %8d  %.2f%%\n',params.channels{cc},h(cc),h(cc)/sum(h)*100);
        end
        fprintf(fid,'-----------------------------------------------------------------\n');
        fprintf(fid,'%40s: %8d  %.2f%%\n','total',sum(h),sum(h)/sum(h)*100);

        if ~isempty(treestats)
            xxx = reshape([treestats{:}],5,[])';
            entropyNorm = max(abs(xxx(:,1)));
            information_gain = max(abs(xxx(:,2)));
        else
            entropyNorm=5;
            information_gain=5;
        end
        f = tree;
        figure(100),clf;

        if params.verbose<5
            warning('not verbose enough, doesn''t plot anything :-)');
        end
        if 1
            c=0;
            for i = 1:length(f.thresholds)
                
                level = floor(log2(i)+1);
                position = i - (2^(level-1)-1);
                child = mod(position-1,2)+1; 
                parent = floor((i-2)/2)+1;

                plevel = floor(log2(parent)+1);
                pposition = parent - (2^(plevel-1)-1);
                pp = find(f.T(:,1)==plevel);
                pp = pp(find(f.T(pp,2)==pposition));
                if parent>0 && (isempty(pp) || f.T(pp,3)<0) %one of parents is leavenode skip this node
                    %%mark children as filled as well for drawing purposes
                    %f.leaves{2*(parent-1)+2} = nan;
                    %f.leaves{2*(parent-1)+3} = nan;
                    continue;
                end

                c=c+1;
                level = f.T(c,1);
                position = f.T(c,2);
                nodefct = f.T(c,3);
                ct = f.T(c,4);
                child = mod(position-1,2)+1; 
                parent = floor(((2^(level-1)-1+position)-2)/2)+1;

                if params.verbose>=5
                    % plot tree
                    parameters=params;
                    if nodefct<0
                        parameters.level=level-1;
                        parameters.position= parent - (2^(level-2)-1);
                    else
                        parameters.level=level;
                        parameters.position = position;
                    end
                    parameters.child = child;
                    if isempty(treestats{i}) 
                        treestats{i} = zeros(1,5);
                    end
                    totaldata = treestats{1}(4)+treestats{1}(3);
                    plot_treePart(100,nodefct,treestats{i}(4),treestats{i}(3),f,parameters,treestats{i}(1),totaldata,treestats{i}(2),entropyNorm,information_gain);
                end
                if params.verbose>=10
                    if nodefct>0
                        disp(f.feats(nodefct,:));
                        if regexp(params.treeHistograms,'sclass','ONCE')
                            disp(histcomb{nodefct});
                        end
                        pause
                    end
                end
            end
        end
        if params.verbose>=5
            axis off
            set(gcf,'Color','white');
            figure(100); print('-depsc',[params.model_name '/fn' sprintf('%03d',fn) '.eps']);
        end

    end

    fprintf(fid,'\n\nSummary for %d trees.\n',params.forestsize);
    for ll = 1:size(channelcounts,3)
        fprintf(fid,'Level %d\n',ll);
            fprintf(fid,'                                                               x1          y1          x2          y2         dist        sx1         sy1          Prop        sx2         sy2          Prop\n');
        mf = mean(mean_fts(cc,fn,ll,:),2);
        sf = mean(std_fts(cc,fn,ll,:),2);
        for cc = 1:length(params.channels)
            fprintf(fid,'%40s: %8d  %7.2f%%',params.channels{cc},sum(channelcounts(cc,:,ll)),sum(channelcounts(cc,:,ll))/sum(sum(channelcounts(:,:,ll)))*100);
            fprintf(fid,'  %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f %5.2f+%5.2f\n', mf(1),sf(1), mf(2),sf(2), mf(3),sf(3), mf(4),sf(4), mf(5),sf(5), mf(6),sf(6), mf(7),sf(7), mf(8),sf(8), mf(9),sf(9), mf(10),sf(10), mf(11),sf(11));
        end
        fprintf(fid,'-----------------------------------------------------------------\n');
        fprintf(fid,'%40s: %8d  %7.2f%%\n','total',sum(sum(channelcounts(:,:,ll))),100);
    end
    fprintf(fid,'\n\nAll levels\n');
    for cc = 1:length(params.channels)
        fprintf(fid,'%40s: %8d  %7.2f%%\n',params.channels{cc},sum(sum(channelcounts(cc,:,:))),sum(sum(channelcounts(cc,:,:)))/sum(channelcounts(:))*100);
    end
    fprintf(fid,'-----------------------------------------------------------------\n');
    fprintf(fid,'%40s: %8d  %7.2f%%\n','total',sum(channelcounts(:)),100);

    fclose(fid);
    system(['cat ' resultsfile]);
end
