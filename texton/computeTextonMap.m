function itx = computeTextonMap(image,clustercenters,flag,params)
  resize = 200;
  w = 3;
  tq = clustercenters;
  img=imread(image);
        itx = zeros(size(img,1),size(img,2));
        %img = imresize(img, [resize resize]); 
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
		 itx(s,r) = l;
                   
             end
        end