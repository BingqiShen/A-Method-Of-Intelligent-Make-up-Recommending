clc;
clear all;

eye_index_xing = xlsread('eye_index_xing.xls');
eye_d_xing = eye_index_xing(:,1);
eye_tan_xing = eye_index_xing(:,2);
eye_innercorner_angle_xing = eye_index_xing(:,3);
eye_outercorner_angle_xing = eye_index_xing(:,4);
plot(eye_innercorner_angle_xing,eye_outercorner_angle_xing,'bx')
% plot3(eye_innercorner_angle_xing,eye_outercorner_angle_xing,eye_d_xing,'bx')         

hold on

% eye_index_danfeng = xlsread('eye_index_danfeng.xls');
% eye_d_danfeng = eye_index_danfeng(:,1);
% eye_tan_danfeng = eye_index_danfeng(:,2);
% eye_r_danfeng = eye_index_danfeng(:,3);
% plot(eye_tan_danfeng,eye_r_danfeng,'rx')         
% 
% hold on

eye_index_taohua = xlsread('eye_index_taohua.xls');
eye_d_taohua = eye_index_taohua(:,1);
eye_arctan_taohua = eye_index_taohua(:,2);
eye_innercorner_angle_taohua = eye_index_taohua(:,3);
eye_outercorner_angle_taohua = eye_index_taohua(:,4);
plot(eye_innercorner_angle_taohua,eye_outercorner_angle_taohua,'rx')
% plot3(eye_innercorner_angle_taohua,eye_outercorner_angle_taohua,eye_d_taohua,'rx')         

hold on

% eye_index_liuye = xlsread('eye_index_liuye.xls');
% eye_d_liuye = eye_index_liuye(:,1);
% eye_tan_liuye = eye_index_liuye(:,2);
% eye_r_liuye = eye_index_liuye(:,3);
% plot(eye_tan_liuye,eye_r_liuye,'bx')        
% 
% hold on

xlabel('eye d index')
ylabel('eye tan index')
% zlabel('face_jaw_regression_index')
grid on
title('ÑÛĞÍÖ¸ÊıÉ¢µãÍ¼')
% legend('ĞÓÑÛ','µ¤·ïÑÛ','ÌÒ»¨ÑÛ','ÁøÒ¶ÑÛ')
% axis([0.68 0.8 0.7 0.9])

