%% 清空环境变量
clc
clear

%% 训练数据预测数据
data=xlsread('nose_data.xlsx');

%从1到349间随机排序，产生随机数，防止过拟合
k=rand(1,97);%产生由在(0, 1)之间均匀分布的随机数组成的数组。
[m,n]=sort(k);%m是排序好的向量，n 是 向量m中对k的索引,这样就将随机数的排列顺序变成整数序号了，即随机抽取的样本


input=data(:,1:4);%输入数据
group=data(:,5);%标签

%随机提取280个样本为训练样本，从训练集和测试集随机选出69个样本为预测样本
input_train=input(n(1:69),:)';%训练样本，要转置
output_train=group(n(1:69),:)';%训练标签，要转置
input_test=input(n(70:97),:)';%测试样本，要转置
output_test=group(n(70:97),:)';%测试标签，要转置

%输入数据归一化
[inputn,inputps]=mapminmax(input_train);

%% BP网络训练
% %初始化网络结构
net=newff(inputn,output_train,10);

net.trainParam.epochs=1000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.0000004;

%% 网络训练
net=train(net,inputn,output_train);

%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);

%网络预测输出
BPoutput=sim(net,inputn_test);%预测标签

%% 结果分析
%根据网络输出找出数据属于哪类
BPoutput(find(BPoutput<1.5))=1;
BPoutput(find(BPoutput>=1.5&BPoutput<2.5))=2;
BPoutput(find(BPoutput>=2.5&BPoutput<3.5))=3;
BPoutput(find(BPoutput>=3.5))=4;
%% 结果分析
%画出预测种类和实际种类的分类图
figure(1)
plot(BPoutput,'og')
hold on
plot(output_test,'r*');
legend('预测类别','输出类别')
title('BP网络预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
ylim([-0.5 4.5])


%预测正确率
rightnumber=0;
for i=1:size(output_test,2)
    if BPoutput(i)==output_test(i)
        rightnumber=rightnumber+1;
    end
end
rightratio=rightnumber/size(output_test,2)*100;

sprintf('测试准确率=%0.2f',rightratio)

w1=net.iw{1,1};
theta1=net.b{1};
w2=net.lw{2,1};
theta2=net.b{2};
