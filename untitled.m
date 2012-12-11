a = zeros(192,256);
for i = 1:256
    for j = 1:192
        a(j,i) = classification(1,256*(j-1)+i);
    end
end
