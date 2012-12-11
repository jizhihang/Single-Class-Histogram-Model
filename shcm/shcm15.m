load('texton-map.mat');

fprintf('Loading image data');
type='bmp';
srcDir='~/texton/shcm/15';               % creating single class histograms for face
files = dir([srcDir '/*.' type]);
result=[];
result15=[];
x=1:1:400;

for i=1:length(files)
        fprintf('%d \n',i);
        if(mod(i, 50) == 0)
            fprintf('.');
        end
        img = imread([srcDir '/' files(i).name]);
        img = imresize(img, [100 100]); 
        [irow icol] = size(img);
        for s=1:irow-2
             fprintf('.');
             for r=1:(icol/3)-2
                 k=img;
                 J = imcrop(img,[r s 2 2]);
                 [a1 a2] = size(J(:,:,1));
                 temp1 = reshape(J(:,:,1),a1*a2,1);
                 temp2 = reshape(J(:,:,2),a1*a2,1);
                 temp3 = reshape(J(:,:,3),a1*a2,1);
                 temp=[temp1 temp2 temp3];
                 [a3 a4] = size(temp);
                 temp = reshape(temp,a3*a4,1);
                 temp=double(temp');
                 l=nearest(temp,tq);
                 result=[result l];
             end
        end
        result15=[result15 result'];
        result=[];
        temp = regexp(files(i).name, '\.', 'split');
        names{i} = temp(1);
end
k=mean(result15,2); 
hist(k,x);
save('resultsforclass15.mat','result15');
% there are various reasns for which i have used the average
% instead of any other calculations.
