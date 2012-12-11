function [tq] = compute_clusters(params)
  k = params.nrTextons;
  fprintf('Loading image data');
  w = 3; % window size wxw patches will be used
  type='bmp';
  srcDir='~/databases/';
  database = 'ms_db';
  ImagesFile = [srcDir database '/UID_train.txt'];
  ImagePath = [srcDir database '/Images/'];

  ImagesList = readTXT2files(ImagesFile,ImagePath);
  resize = 200;
  pixels = (resize-2)*(resize-2)*length(ImagesList);
  data = [];
  names = {};
  tmp = zeros(27,1);


  data2=zeros(27,pixels);
  files = dir([srcDir '/*.' type]);

  for i=1:length(ImagesList)
        if(mod(i, 50) == 0)
            fprintf('\n');
        end
        img = imread(char(ImagesList(i)));
	char(ImagesList(i))
        img = imresize(img, [resize resize]); 
        [irow icol] = size(img);
        %fprintf('.');
        for s=2:(irow-1)
             for r=2:((icol)/3-1)
                 J = imcrop(img,[r-1 s-1 w-1 w-1]);
                 [a1 a2] = size(J(:,:,1));
                 temp1 = reshape(J(:,:,1),a1*a2,1);
                 temp2 = reshape(J(:,:,2),a1*a2,1);
                 temp3 = reshape(J(:,:,3),a1*a2,1);
                 temp=[temp1 temp2 temp3];
                 [a3 a4] = size(temp);
                 temp = reshape(temp,a3*a4,1);
                 %index = idivide(i,int8(10))+1;
                 index = (i-1)*(resize-2)*(resize-2)+(s-2)*(resize-2)+(r-1);
                 data2(:,index) = temp;
                 
                 if(mod(index, 1000) == 0)
                    fprintf('.');
                 end
                 
             end
        end
        %temp = regexp(files(i).name, '\.', 'split');
        %names{i} = temp(1);
end

save('images_train.mat','data2','-v7.3');

t1 = toc % time to load images

%load('images_train.mat')

%t1 = toc % time to load images

tic

data2=data2';

data2=double(data2);
ran = randi(pixels,1,10000);
[tu,tq]=kmeans(data2(ran,:),k,'emptyaction','drop');
