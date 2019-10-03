function [ index ] = crossvali( bins )
    for 1:bins
        numBins = bins; % Number of Bins you want to devide your data into
        numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
        selectAtRandom = true; % true = select features at random, false = select the first features

        [ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

        b=zeros(1,bins);
        for k = 1:bins
            a=kNN(Xt{2}, k, Xt{1}, Lt{1});
            conf = calcConfusionMatrix(a, Lt{2});
            cvscore=calcAccuracy(conf);
            b(k,1)=cvscore;
        end  
        [~, index]=max(b);
        double(max(b))
    end
end

%cross=crossval(LkNN,X);
%acc=1-kfoldloss(cross)
