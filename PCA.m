X=load('origin_data.txt');

x=zscore(X);                       %��׼��

[coef,score,eig,t]=pca(x);   %����pca�������

t;                                %ÿһ���������������µ�ԭ��ľ���

s=0;

i=1;

while s/sum(eig)<0.95

    s=s+eig(i);

    i=i+1;

end                              %����ۼƹ����ʴ���95%��������
i = i-1;
NEW=x*coef;
NEW=NEW(:,1:i);              %����µ�����

figure

pareto(eig/sum(eig));          %���������ֱ��ͼ