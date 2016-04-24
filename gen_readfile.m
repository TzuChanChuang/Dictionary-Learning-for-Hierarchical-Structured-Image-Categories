fileID = fopen('all.txt','w');
for i=1:2
    for j= 1:10
        fprintf(fileID,'%f ',j);
    end
end

fprintf(fileID,'%6s %12s\n','x','exp(x)');
fprintf(fileID,'%6.2f %12.8f\n',A);
fclose(fileID);