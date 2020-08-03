clc;
clear all;

train_num = 85;
%读取excel文件
face_index_edan = xlsread('face_index_edan.xls');
face_index_edan = [face_index_edan(:,1),face_index_edan(:,3)];
face_index_guazi = xlsread('face_index_guazi.xls');
face_index_guazi = [face_index_guazi(:,1),face_index_guazi(:,3)];
%输入训练数据
face_train_data = [face_index_edan(1:train_num,:);face_index_guazi(1:train_num,:)];
face_train_judge1 = ones(1,train_num)';
face_train_judge2 = zeros(1,train_num)';
face_train_judge = [face_train_judge1;face_train_judge2];
%输入测试数据
face_test_data = [face_index_edan(train_num+1:length(face_index_edan),:);face_index_guazi(train_num+1:length(face_index_guazi),:)];
%进行线性逻辑回归
[testNum, attrNum] = size(face_test_data);
face_testdata2 = [ones(testNum,1), face_test_data];
B = glmfit(face_train_data, [face_train_judge,ones(size(face_train_judge))],'binomial', 'link', 'logit');
p = 1.0 ./ (1 + exp(- face_testdata2 * B));
%画出散点图
face_shape_index_edan = face_index_edan(:,1);
face_jaw_r_index_edan = face_index_edan(:,2);  
plot(face_shape_index_edan,face_jaw_r_index_edan,'rx')
hold on

face_shape_index_guazi = face_index_guazi(:,1);
face_jaw_r_index_guazi = face_index_guazi(:,2);
plot(face_shape_index_guazi,face_jaw_r_index_guazi,'bx')

xlabel('face shape index')
ylabel('face jaw r index')
grid on
title('脸部指数散点图')
legend('鹅蛋脸','瓜子脸')
%画出回归曲线
hold on
x = 0.7:0.001:0.88;   
y = (B(2,1)/-B(3,1)).*x+B(1,1)/-B(3,1);
plot(x,y)

%计算正误率
true_num = 0;
for i = 1:length(face_index_edan)-length(face_train_judge1)
    if(p(i)>=0.5)
        true_num = true_num + 1;
    end
end

for i = 1:length(face_index_guazi)-length(face_train_judge2)
    if(p(i+length(face_index_edan)-length(face_train_judge1))<0.5)
        true_num = true_num + 1;
    end
end

true_rate = true_num/length(p);
false_rate = 1 - true_rate;
T = ['正确率：',num2str(true_rate)];
F = ['错误率：',num2str(false_rate)];
disp('使用Logistic regression后：')
disp(T)
disp(F)


