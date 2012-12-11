function l=nearest(a,b)
    [wa ,wb]=size(b);
    A2=a;
    B2=b(1,:);
    req=1;
    min=sqrt(dot(A2-B2,A2-B2));
    for i=1:wa
        temp=b(i,:);
        dist=sqrt(dot(A2-temp,A2-temp));
        if dist < min
            req=i;
            min=dist;
        end
    end
    l=req;
end