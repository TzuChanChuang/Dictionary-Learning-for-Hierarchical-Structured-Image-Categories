load('catndog_3_1000.mat');
cat_700 = [cat_3_1000(:,1:700) cat_3_1000(:,1001:1700) cat_3_1000(:,2001:2700)];
cat_700_t = [cat_3_1000_t(:,1:700) cat_3_1000_t(:,1001:1700) cat_3_1000_t(:,2001:2700)];
cat_700_test = [cat_3_1000(:,701:1000) cat_3_1000(:,1701:2000) cat_3_1000(:,2701:3000)];
cat_700_test_t = [cat_3_1000_t(:,701:1000) cat_3_1000_t(:,1701:2000) cat_3_1000_t(:,2701:3000)];

dog_700 = [dog_3_1000(:,1:700) dog_3_1000(:,1001:1700) dog_3_1000(:,2001:2700)];
dog_700_t = [dog_3_1000_t(:,1:700) dog_3_1000_t(:,1001:1700) dog_3_1000_t(:,2001:2700)];
dog_700_test = [dog_3_1000(:,701:1000) dog_3_1000(:,1701:2000) dog_3_1000(:,2701:3000)];
dog_700_test_t = [dog_3_1000_t(:,701:1000) dog_3_1000_t(:,1701:2000) dog_3_1000_t(:,2701:3000)];

save('catndog_700.mat','cat_700','cat_700_t','cat_700_test','cat_700_test_t','dog_700','dog_700_t','dog_700_test','dog_700_test_t');