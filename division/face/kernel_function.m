clc;
clear all;

face_index_edan = xlsread('face_index_edan.xls');
face_index_edan = [face_index_edan(:,1),face_index_edan(:,3)];
% face_shape_index_edan_guazi = face_index_edan_guazi(:,1);
% face_jaw_index_edan_guazi = face_index_edan_guazi(:,2);  
% plot(face_shape_index_edan_guazi,face_jaw_index_edan_guazi,'rx')
% 
% hold on

face_index_guazi = xlsread('face_index_guazi.xls');
face_index_guazi = [face_index_guazi(:,1),face_index_guazi(:,3)];
% face_shape_index_yuan_fang = face_index_yuan_fang(:,1);
% face_jaw_index_yuan_fang = face_index_yuan_fang(:,2);
% plot(face_shape_index_yuan_fang,face_jaw_index_yuan_fang,'bx')

% xlabel('face shape index')
% ylabel('face jaw index')
% grid on
% title('脸部指数散点图')
% legend('鹅蛋脸+瓜子脸','圆脸+方脸')

train_num = 85;

train_data = [face_index_edan(1:train_num,:);face_index_guazi(1:train_num,:)];
train_group1 = ones(1,train_num)';
train_group2 = ones(1,train_num)'*(-1);
train_groups=[train_group1;train_group2];

test_data = [face_index_edan(train_num+1:length(face_index_edan),:);face_index_guazi(train_num+1:length(face_index_guazi),:)];
test_group1 = ones(1,length(face_index_edan)-length(train_group1))';
test_group2 = ones(1,length(face_index_guazi)-length(train_group2))'*(-1);
test_group = [test_group1;test_group2];
svmModel = svmtrain(train_data,train_groups,'kernel_function','rbf','showplot',true);
classification=svmclassify(svmModel,test_data,'Showplot',true);
xlabel('face shape index')
ylabel('face jaw  r index')
grid on
%计算正误率
true_num = 0;
for i = 1:length(test_group1)
    if(classification(i)==1)
        true_num = true_num + 1;
    end
end

for i = 1:length(test_group2)
    if(classification(i+length(test_group1))==-1)
        true_num = true_num + 1;
    end
end

true_rate = true_num/length(classification);
false_rate = 1 - true_rate;
T = ['正确率：',num2str(true_rate)];
F = ['错误率：',num2str(false_rate)];
disp('使用Kernel function后：')
disp(T)
disp(F)

