
%% Metrics for assessing flexibility


    data=spike_array; 
    time = linspace(0,size(data,2)/20000,size(data,2));
    smoothedDR = zeros(size(data));
    w = hann(round(0.4*20000));
    w = w/sum(w);
    for i = 1:size(data,1)
        smoothedDR(i,:) = conv(data(i,:),w,'same')/(1/20000);
    end
    % with this we get the smoothed data and now we proceed to downsample

    time = downsample(time',400).';
    smoothedDR = downsample(smoothedDR',400).';
    smoothedDR2 = smoothedDR;
    smoothedDR2(smoothedDR2==0) = NaN;

    %run all combinations
    
    ncomb = nchoosek(1:1:size(spike_array,1),2);
    displacement = zeros(1,size(ncomb,1));
    different_biophysical_properties = zeros(1,size(ncomb,1));

    for i = 1:size(ncomb,1)
        displacement(i) = max(mudisplacement({smoothedDR([ncomb(i,:)],:)}, 20000));
        different_biophysical_properties(i)=(VthCa_newTA(ncomb(i,1))- VthCa_newTA(ncomb(i,2)));
    end





  
