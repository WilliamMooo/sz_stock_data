X=load('origin_data.txt');

x=zscore(X);                       %标准化

[coef,score,eig,t]=pca(x);   %利用pca处理矩阵

t;                                %每一组数据在新坐标下到原点的距离

s=0;

i=1;

while s/sum(eig)<0.95

    s=s+eig(i);

    i=i+1;

end                              %获得累计贡献率大于95%几组数据
i = i-1;
NEW=x*coef;
NEW=NEW(:,1:i);              %输出新的数据

figure

pareto(eig/sum(eig));          %输出贡献率直方图