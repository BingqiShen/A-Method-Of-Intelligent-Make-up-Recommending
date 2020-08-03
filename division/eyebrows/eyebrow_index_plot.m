clc;
clear all;

eyebrow_biao_index = xlsread('eyebrow_biao_index.xls');
eyebrow_area_biao = eyebrow_biao_index(:,1);
eyebrow_zhouchang_biao = eyebrow_biao_index(:,2);
eyebrow_width_biao = eyebrow_biao_index(:,3);
eyebrow_r_biao = eyebrow_biao_index(:,4);
plot(eyebrow_area_biao,eyebrow_r_biao,'bx')
% plot3(eye_innercorner_angle_xing,eye_outercorner_angle_xing,eye_d_xing,'bx')         

hold on

eyebrow_yizi_index = xlsread('eyebrow_yizi_index.xls');
eyebrow_area_yizi = eyebrow_yizi_index(:,1);
eyebrow_zhouchang_yizi = eyebrow_yizi_index(:,2);
eyebrow_width_yizi = eyebrow_yizi_index(:,3);
eyebrow_r_yizi = eyebrow_yizi_index(:,4);
plot(eyebrow_area_yizi,eyebrow_r_yizi,'rx')
% plot3(eye_innercorner_angle_xing,eye_outercorner_angle_xing,eye_d_xing,'bx')         

hold on

eyebrow_liuye_index = xlsread('eyebrow_liuye_index.xls');
eyebrow_area_liuye = eyebrow_liuye_index(:,1);
eyebrow_zhouchang_liuye = eyebrow_liuye_index(:,2);
eyebrow_width_liuye = eyebrow_liuye_index(:,3);
eyebrow_r_liuye = eyebrow_liuye_index(:,4);
plot(eyebrow_area_liuye,eyebrow_r_liuye,'gx')
% plot3(eye_innercorner_angle_xing,eye_outercorner_angle_xing,eye_d_xing,'bx')         

hold on

eyebrow_jian_index = xlsread('eyebrow_jian_index.xls');
eyebrow_area_jian = eyebrow_jian_index(:,1);
eyebrow_zhouchang_jian = eyebrow_jian_index(:,2);
eyebrow_width_jian = eyebrow_jian_index(:,3);
eyebrow_r_jian = eyebrow_jian_index(:,4);
plot(eyebrow_area_jian,eyebrow_r_jian,'kx')
% plot3(eye_innercorner_angle_xing,eye_outercorner_angle_xing,eye_d_xing,'bx')         

hold on

xlabel('eyebrow area index')
ylabel('eyebrow r index')
% zlabel('face_jaw_regression_index')
grid on
title('眉型指数散点图')
legend('标准眉','一字眉','柳叶眉','剑眉')
% axis([0.68 0.8 0.7 0.9])

