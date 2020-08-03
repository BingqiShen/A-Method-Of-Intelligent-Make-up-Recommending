clc;
clear all;

face_index_edan = xlsread('face_index_edan.xls');
face_shape_index_edan = face_index_edan(:,1);
face_jaw_index_edan = face_index_edan(:,2);
face_jaw_r_index_edan = face_index_edan(:,3);   
plot(face_shape_index_edan,face_jaw_index_edan,'rx')          % face_jaw_index_edan,

hold on

face_index_fang = xlsread('face_index_fang.xls');
face_shape_index_fang = face_index_fang(:,1);
face_jaw_index_fang = face_index_fang(:,2);
face_jaw_regression_index_fang = face_index_fang(:,3);
plot(face_shape_index_fang,face_jaw_index_fang,'bx')  % face_jaw_index_fang,

hold on

face_index_guazi = xlsread('face_index_guazi.xls');
face_shape_index_guazi = face_index_guazi(:,1);
face_jaw_index_guazi = face_index_guazi(:,2);
face_jaw_r_index_guazi = face_index_guazi(:,3);
plot(face_shape_index_guazi,face_jaw_index_guazi,'gx')  %,face_jaw_index_guazi,

hold on

face_index_yuan = xlsread('face_index_yuan.xls');
face_shape_index_yuan = face_index_yuan(:,1);
face_jaw_index_yuan = face_index_yuan(:,2);
face_jaw_regression_index_yuan = face_index_yuan(:,3);
plot(face_shape_index_yuan,face_jaw_index_yuan,'kx')  %,  face_jaw_index_yuan,

hold on;
% x = 0.7:0.001:0.91;   
% y = (-13.7958/100.5662).*x+92.9564/100.5662;
% plot(x,y)
xlabel('face shape index')
ylabel('face jaw  index')
% zlabel('face_jaw_regression_index')
grid on
title('Á³²¿Ö¸ÊýÉ¢µãÍ¼')
legend('¶ìµ°Á³','·½Á³','¹Ï×ÓÁ³','Ô²Á³')
% axis([0.68 0.8 0.7 0.9])

