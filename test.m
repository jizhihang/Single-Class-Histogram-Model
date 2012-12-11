 %nclasses=params.nrClasses;
        %treeparams = {params.nrClasses,params.maxLevel,params.stopentropy,params.max_gain,params.nrFeatsPerNode,bitand(params.normalizeByPrior,1),imageTypes,imageScales,nrNonZeroChannels,params.nr_samples,params.nrThreshOpt};
        k = 400;
        nclass = 2;
        hist = zeros(k,1);
        nhists = zeros(k,nclass);
        
        for i = 1:nclass
	  loadfile = sprintf('results/resultsforclass%d.mat',i);
	  load(loadfile);
	  x = size(result1,1);
	  y = size(result1,2);
	  hvector = reshape(result1,1,x*y);
	  for j = 1:x*y
	    hist(hvector(1,j)) = hist(hvector(1,j))+1;
	    end
	    
	  nhists(:,i) = hist; 
	  end
	  
	  
  
        
        load('images.mat');
        histograms=[];
        fprintf('Now go into pdDecTree...\n');
        fprintf('6\n');
        [tree,leaves,thresholds,treestats]=pdDecTree(intI,gt,pixeldiffs,treeparams,nhists);

        %fprintf('%d pixeldiffs for %d images: %.2fs\n\n',params.ft_pertree,length(params.ImagesList),toc);
    
        %tree=convertTree(pixeldiffs,nhists,tree,leaves,thresholds,treestats,params);
    
        save( '001.mat','tree','treestats');