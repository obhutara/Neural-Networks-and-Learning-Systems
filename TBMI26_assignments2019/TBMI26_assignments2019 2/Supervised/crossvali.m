b=zeros(100,1);
for k = 1:100
    a=kNN(Xt{2}, k, Xt{1}, Lt{1});
    conf = calcConfusionMatrix(a, Lt{2});
    cvscore=calcAccuracy(conf);
    b(k,1)=cvscore;
end  

[~, index]=max(b);
index
double(max(b))

%cross=crossval(LkNN,X);
%acc=1-kfoldloss(cross)
