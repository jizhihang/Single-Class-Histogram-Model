tic

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

load('images_train.mat');

t1 = toc % time to load images

%load('images_train.mat')

%t1 = toc % time to load images

tic

data2=data2';
k=30;
data2=double(data2);
ran = randi(pixels,1,10000);
[tu,tq]=kmeans(data2(ran,:),k);

save('texton-map.mat','tu','tq');

t2 = toc; %time to find kmeans

tic;

R = random_color(k,'Pastel');

for i=1:length(ImagesList)
        if(mod(i, 50) == 0)
            fprintf('\n');
        end
        img = imread(char(ImagesList(i)));
        %imshow(img);
        img2 = zeros(resize,resize,3);
        img = imresize(img, [resize resize]); 
        [irow icol] = size(img);
        fprintf('.');
        for s=2:(irow-1)
             for r=2:((icol/3)-1)
                
                 J = imcrop(img,[r-1 s-1 w-1 w-1]);
                 [a1 a2] = size(J(:,:,1));
                 temp1 = reshape(J(:,:,1),a1*a2,1);
                 temp2 = reshape(J(:,:,2),a1*a2,1);
                 temp3 = reshape(J(:,:,3),a1*a2,1);
                 temp=[temp1 temp2 temp3];
                 [a3 a4] = size(temp);
                 temp = reshape(temp,a3*a4,1);
                 temp=double(temp');
                 l=nearest(temp,tq);
		 img2(s,r,1) = R(l,1);
		 img2(s,r,2) = R(l,2);
		 img2(s,r,3) = R(l,3);
                   
             end
        end
        
         
        file = regexp(char(ImagesList(i)), '/', 'split');
        name = regexp(file(length(file)),'\.','split');
        fname = name{1}(1);
        
        %name = file(regexp(length(file),'\.','split'));
        hold on;
        h = imshow(img2);
        
        saveas(h, strcat(char(fname),'.jpg'));
        hold off;
end

t3 = toc; % time to color
