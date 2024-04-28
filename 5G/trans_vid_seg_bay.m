function [delay, recv, bler_all, ratio, thr, thr_snr, bler, snr_me, cqi_me, blk] = trans_vid_seg_bay (vid_path, pdsch, channel, carrier)  

%% Channel transition matrix (first-order Markov chain)
step = 0.1;
beta_list = [5, 5.01, 0.84, 1.67, 1.61, 1.64, 3.87, 5.06, 6.4, 12.6, 17.6, 23.3, 29.5, 33.0, 35.4];
CQ_var = 0.2:step:8;
discrete_values = CQ_var;
h_tran = [];
for i =1:length(CQ_var) 
    m = CQ_var(i);
    rou = 0.9;
    pdf_continuous = @(x) 1/(1-rou^2)*(x/(rou^2*m)).^2.*exp(-(rou^2*x+m)/(1-rou^2)).*besseli(2, rou*2*sqrt(m*x)/(1-rou^2));
    discrete_values = CQ_var;
    discrete_pdf = pdf_continuous(discrete_values) * step;
    discrete_pdf = discrete_pdf / sum(discrete_pdf);
    h_tran = [h_tran; discrete_pdf];
end
soft = 1;

%% Channel KPIs
snr = 0;
thr_snr = [];
thr = [];
th = 0;
err = 0;
bler = [];
snr_me = [];
cqi_me = [];
blk = [];
video = vid_path;
%CQ_var = 10.^((-5:14)/10);  %SNR ratio
beta_1 =  -1.7429;
beta_2 = 1.33;
Rk = [0.15, 0.23, 0.37, 0.60, 1.69, 2.4, 3.02, 3.3, 3.9, 4.5, 5.1, 5.33, 6.22, 6.91, 7.40];

%% Simulation parameters for physical channel
%soft_ack_true = 1;   % enable softACK
state_buffer = zeros(5,2); 
pay_load = 0.8;

log_ = [];   % log the number of transmisssion slots, which in turn is utilized for delay calculation

%% Simulation Parameters
rou = besselj(0,2*pi*channel.MaximumDopplerShift*0.001);
mcs = [2,78;2,120;2,193;2,308;4,449;4,616;6,378;6,567;6,666;6,772;6,873;8,682.5;8,797;8,885;8,948];
pdf_values = exp(-(1:15));  
mcs_p = pdf_values / sum(pdf_values);   % probability distribution of the mcs

perfectEstimation = false; % Perfect synchronization and channel estimation
%random_number = randi([1, 1000]);
rng('shuffle');            % Set default random number generator for repeatability
%img = './dataset/peppers.png'; % Test image
%carrier = nrCarrierConfig;  % Carrier configuration  default: 15kHz scs, 52 grid size 
show_cons = false;          % Show constellation diagram
temp = exp(-(1:length(CQ_var)));
CQ_profile = temp/sum(temp);   % initialize the pdf of channel quality


%% CQI SNR initialization
SNRdB = 5;   % avg snr
amp_noise = 1/(10^(SNRdB/20));


% %% PDSCH and DL-SCH configuration
% pdsch = nrPDSCHConfig;
% pdsch.NumLayers = 2;   % Set according to DCI RI
% %pdsch.PRBSet = 0:(carrier.NSizeGrid-1);     % Full band allocation
% pdsch.PRBSet = 0:5;
% 
% 
% % Set DM-RS to improve channel estimation
% pdsch.DMRS.DMRSAdditionalPosition = 1;
% pdsch.DMRS.DMRSConfigurationType = 1;
% pdsch.DMRS.DMRSLength = 2;
% %pdsch.DMRS                            % Display DM-RS properties

% HARQ
NHARQProcesses = 16;     % Number of parallel HARQ processes
rvSeq = [0 2 3 1]; % Set redundancy version vector
% rvSeq = 0; % No harq
% HARQ management
harqEntity = HARQEntity(0:NHARQProcesses-1,rvSeq,pdsch.NumCodewords);  % Process, redundancy version, codewords

 % Create DL-SCH encoder object
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = true;


% Create DL-SCH decoder object
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = true;
% LDPC parameters
decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
decodeDLSCH.MaximumLDPCIterationCount = 10;


%% Channel Modeling
nTxAnts = 2;   % Number of transmit antennas
nRxAnts = 2;   % Number of receive antennas
% 
% % Check that the number of layers is valid for the number of antennas
% if pdsch.NumLayers > min(nTxAnts,nRxAnts)
%     error("The number of layers ("+string(pdsch.NumLayers)+") must be smaller than min(nTxAnts,nRxAnts) ("+string(min(nTxAnts,nRxAnts))+")")
% end
% 
% % Create a channel object
% channel = nrTDLChannel;
% channel.DelayProfile = "TDL-C";    % Tapped delay line channel, C type
% channel.MaximumDopplerShift = do_shift;
% channel.NumTransmitAntennas = nTxAnts;
% channel.NumReceiveAntennas = nRxAnts;
% 
% Set the channel sample rate to that of the OFDM signal
% ofdmInfo = nrOFDMInfo(carrier);
% channel.SampleRate = ofdmInfo.SampleRate;
% 
% % Channel constellation diagram
% constPlot = comm.ConstellationDiagram;                                          % Constellation diagram object
% constPlot.ReferenceConstellation = getConstellationRefPoints(pdsch.Modulation); % Reference constellation values
% constPlot.EnableMeasurements = 1;                                               % Enable EVM measurements
% 
 % Initial timing offset
offset = 0;

estChannelGrid = getInitialChannelEstimate(channel,carrier); % Channel estimation
newPrecodingWeight = getPrecodingMatrix(pdsch.PRBSet,pdsch.NumLayers,estChannelGrid);  % Precoding matrix, set according to DCI

[tx, vid_len] = read_video(video);    % read video stream
tx_tb = [tx;zeros(50000*1500,1)];   % padding zeros behind the video for retransmission

recv = [{}];     % Receive signal, a cell array
bler = [];           % Record BLER
bler_tmp = [];    % tem_bler for bandwidth calculation
nSlot = 0;

%doppler_chge = 30;  % doppler variations per 30 slots

vid_track = 0;
end_flag = 0;
cnt_down = NHARQProcesses*3;  % count down of the transmission when the last slot has been sent


while true
%     if mod(nSlot,doppler_chge)
%         channel.MaximumDopplerShift = normrnd(10,4,1,1);      % sample the doppler frequency shift from Gaussian distribution periodically
%     end
    % select mcs according to current cqi
    
    mcs_k = randsample(1:15, 1, true, mcs_p);
    mo_order = mcs(mcs_k,1);
    switch mo_order
        case 2
            pdsch.Modulation = "QPSK";
        case 4
            pdsch.Modulation = "QPSK";
        case 6
            pdsch.Modulation = "QPSK";
        case 8
            pdsch.Modulation = "QPSK";
    end
    if pdsch.NumCodewords == 1   
        codeRate = mcs(mcs_k,2)/1024;
    else
        codeRate = [mcs(mcs_k,2) mcs(mcs_k,2)]./1024;
    end
    encodeDLSCH.TargetCodeRate = codeRate;
    decodeDLSCH.TargetCodeRate = codeRate;

    %% Use the transition probability to predict the channel quality profile
    CQ_profile = CQ_profile*h_tran;

    %% Calculate the avg BLER of MCS k
    bler_mcs = zeros(1,15);
    for i = 1:length(mcs)
       for j = 1:length(CQ_var)
           bler_mcs(i) = bler_mcs(i) + 1/(1+exp(-beta_1*CQ_var(j)-beta_2*i+1.2477))*CQ_profile(j);
       end
    end

    %% determine the optimized CQI distribution
   Rk = [0.15, 0.23, 0.37, 0.60, 1.69, 2.4, 3.02, 3.3, 3.9, 4.5, 5.1, 5.33, 6.22, 6.91, 7.40];

    % 定义目标函数，其中p是优化变量，rk、nk是已知参数
    objective = @(p) -sum(p .* Rk .* (1 - bler_mcs));
    
    % 定义约束
    Aeq = ones(1, 15);  % 确保p(k)的和等于1
    beq = 1;
    
    % 定义约束
    lb = zeros(1, 15);  % p(k)的下限为0
    ub = ones(1, 15);   % p(k)的上限为1
    
    % 添加额外的约束，所有的 p(k) * n(k) 的总和小于阈值
    A = bler_mcs;  % 这里只有一行，因为是所有约束总和
    b = 0.1;  % 阈值
    
    % 初始点
    x0 = ones(1, 15) / 15;  % 可以选择一个合适的初始点
    
    % 优化
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxIter', 1000);  % 将MaxIter设置为较大的值，例如1000，并关闭显示信息
    [mcs_p, fval] = fmincon(objective, x0, A, b, Aeq, beq, lb, ub, [], options);
    mcs_p = (mcs_p - min(mcs_p)) / (max(mcs_p) - min(mcs_p));
    mcs_p  = mcs_p / sum(mcs_p);


    % New slot
    carrier.NSlot = nSlot;
    % Generate PDSCH indices info, which is needed to calculate the transport
    % block size
    [pdschIndices,pdschInfo] = nrPDSCHIndices(carrier,pdsch);

    % Calculate transport block sizes
    Xoh_PDSCH = 0;
    trBlkSizes = nrTBS(pdsch.Modulation,pdsch.NumLayers,numel(pdsch.PRBSet),pdschInfo.NREPerPRB,codeRate,Xoh_PDSCH);
    

    %% Get new transport blocks and flush decoder soft buffer, as required
    for cwIdx = 1:pdsch.NumCodewords
        if harqEntity.NewData(cwIdx)   % check whether this is a new data; start of the redundancy version
            % Create and store a new transport block for transmission
            if vid_track+trBlkSizes >= length(tx_tb)
               trBlk = zeros(trBlkSizes,1);
               trBlk(1:length(tx_tb)-vid_track) = tx_tb(vid_track+1:length(tx_tb));  % padding zeros
               end_flag = 1;
            else
               trBlk = tx_tb(vid_track+1:vid_track+trBlkSizes);    
            end
            setTransportBlock(encodeDLSCH,trBlk,cwIdx-1,harqEntity.HARQProcessID);

            % If the previous RV sequence ends without successful
            % decoding, flush the soft buffer
            if harqEntity.SequenceTimeout(cwIdx)
                resetSoftBuffer(decodeDLSCH,cwIdx-1,harqEntity.HARQProcessID);
            end
        end
    end
    codedTrBlock = encodeDLSCH(pdsch.Modulation,pdsch.NumLayers,pdschInfo.G,harqEntity.RedundancyVersion,harqEntity.HARQProcessID);


    th = th+trBlkSizes;  % accumulate the size of transport block for throughput estimation
    blk = [blk, trBlkSizes];


    %% PDSCH Modulation and MIMO
    pdschSymbols = nrPDSCH(carrier,pdsch,codedTrBlock);
    % DM-RS symbols generation
    dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
    precodingWeights = newPrecodingWeight;
    pdschSymbolsPrecoded = pdschSymbols*precodingWeights;

    % Resource grid mapping
    pdschGrid = nrResourceGrid(carrier,nTxAnts);
    [~,pdschAntIndices] = nrExtractResources(pdschIndices,pdschGrid);
    pdschGrid(pdschAntIndices) = pdschSymbolsPrecoded;

    % PDSCH DM-RS precoding and mapping
    for p = 1:size(dmrsSymbols,2)
        [~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),pdschGrid);
        pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols(:,p)*precodingWeights(p,:);
    end

    [txWaveform,waveformInfo] = nrOFDMModulate(carrier,pdschGrid); % OFDM Modulation

    %% Pass thru TDL channel
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
    txWaveform = [txWaveform; zeros(maxChDelay,size(txWaveform,2))];  % Padding zeros for delay flush
    
    [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
    if ~mod(length(log_), 100)
        %SNRdB = randn*4+6;
        noise = generateAWGN(SNRdB,nRxAnts,waveformInfo.Nfft,size(rxWaveform));
    end
    noise = generateAWGN(SNRdB,nRxAnts,waveformInfo.Nfft,size(rxWaveform));
    rxWaveform = rxWaveform + noise;

    %% Perform perfect or practical timing estimation and synchronization
    if perfectEstimation
        % Get path filters for perfect timing estimation
        pathFilters = getPathFilters(channel); 
        [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
    else
        [t,mag] = nrTimingEstimate(carrier,rxWaveform,dmrsIndices,dmrsSymbols);
        offset = hSkipWeakTimingOffset(offset,t,mag);
    end
    rxWaveform = rxWaveform(1+offset:end,:);

    %% Demodulation, channel estimation
    rxGrid = nrOFDMDemodulate(carrier,rxWaveform);   % OFDM-demodulate the synchronized signal
    
    if perfectEstimation
        % Perform perfect channel estimation between transmit and receive
        % antennas.
        estChGridAnts = nrPerfectChannelEstimate(carrier,pathGains,pathFilters,offset,sampleTimes);

        % Get perfect noise estimate (from noise realization)
        noiseGrid = nrOFDMDemodulate(carrier,noise(1+offset:end ,:));
        noiseEst = var(noiseGrid(:));

        % Get precoding matrix for next slot
        newPrecodingWeight = getPrecodingMatrix(pdsch.PRBSet,pdsch.NumLayers,estChGridAnts);

        % Apply precoding to estChGridAnts. The resulting estimate is for
        % the channel estimate between layers and receive antennas.
        estChGridLayers = precodeChannelEstimate(estChGridAnts,precodingWeights.');
    else
        % Perform practical channel estimation between layers and receive
        % antennas.
        [estChGridLayers,noiseEst] = nrChannelEstimate(carrier,rxGrid,dmrsIndices,dmrsSymbols,'CDMLengths',pdsch.DMRS.CDMLengths);

        % Remove precoding from estChannelGrid before precoding
        % matrix calculation
        estChGridAnts = precodeChannelEstimate(estChGridLayers,conj(precodingWeights));

        % Get precoding matrix for next slot
        newPrecodingWeight = getPrecodingMatrix(pdsch.PRBSet,pdsch.NumLayers,estChGridAnts);
    end

%% Plot channel estimate between the first layer and the first receive antenna
%        mesh(abs(estChGridLayers(:,:,1,1)));  
%        title('Channel Estimate');
%        xlabel('OFDM Symbol');
%        ylabel("Subcarrier");
%        zlabel("Magnitude");

        %% Equalization
        [pdschRx,pdschHest] = nrExtractResources(pdschIndices,rxGrid,estChGridLayers);
        [pdschEq,csi] = nrEqualizeMMSE(pdschRx,pdschHest,noiseEst);

        %% Constellation diagram
        if show_cons
         constPlot.ChannelNames = "Layer "+(pdsch.NumLayers:-1:1);
         constPlot.ShowLegend = true;
         % Constellation for the first layer has a higher SNR than that for the
         % last layer. Flip the layers so that the constellations do not mask
         % each other.
         constPlot(fliplr(pdschEq));
        end
         
        %% PDSCH and DL-SCH Decode
        [dlschLLRs,rxSymbols] = nrPDSCHDecode(carrier,pdsch,pdschEq,noiseEst);
         % Scale LLRs by CSI
        csi = nrLayerDemap(csi);                                    % CSI layer demapping
        for cwIdx = 1:pdsch.NumCodewords
            Qm = length(dlschLLRs{cwIdx})/length(rxSymbols{cwIdx}); % Bits per symbol
            csi{cwIdx} = repmat(csi{cwIdx}.',Qm,1);                 % Expand by each bit per symbol
            dlschLLRs{cwIdx} = dlschLLRs{cwIdx} .* csi{cwIdx}(:);   % Scale
        end
        csi_arr = cell2mat(csi);
        csi_arr = csi_arr(1,:);


        
        if harqEntity.TransmissionNumber == 0     % modified
            decodeDLSCH.TransportBlockLength = trBlkSizes;  
            %acc_tbs = acc_tbs + trBlkSizes;
        else
            decodeDLSCH.TransportBlockLength = harqEntity.TransportBlockSize;
            %acc_tbs = acc_tbs + harqEntity.TransportBlockSize;
        end

        [decbits,blkerr] = decodeDLSCH(dlschLLRs,pdsch.Modulation,pdsch.NumLayers, ...
        harqEntity.RedundancyVersion,harqEntity.HARQProcessID);

        beta = beta_list(mcs_k);
        snr_list = abs(csi_arr)/amp_noise;
        snr_avg = beta*(log(1/length(csi_arr)*sum(exp(snr_list/beta))));
        snr_avg = beta*(log(1/length(csi_arr)*sum(exp(snr_list/beta))));
        pr = 1/(1+(exp(1.7429*snr_avg-1.3*mcs_k+1.2477)));

        if soft
           if pr < 0.6
               blkerr = 1;
           end
        end
        
        %% update posterior probability
        if blkerr
            CQ_profile = (1./(1+(exp(-beta_1*(CQ_var)-beta_2*(ones(1,length(CQ_var))*mcs_k)+1.2477)))/bler_mcs(mcs_k)).*CQ_profile;
        else
            CQ_profile = ((1-1./(1+(exp(-beta_1*(CQ_var)-beta_2*(ones(1,length(CQ_var))*mcs_k)+1.2477))))/(1-bler_mcs(mcs_k))).*CQ_profile;
        end
        err = err+blkerr; % accumulate the bloc error for throughput estimation
        bler = [bler, blkerr];
    

        %% HARQ report
        if harqEntity.TransmissionNumber == 0   %initial transmission
        %if harq_state(harqEntity.HARQProcessID+1,2) == 0
           statusReport = updateAndAdvance(harqEntity,blkerr,trBlkSizes,pdschInfo.G);
        else
           statusReport = updateAndAdvance(harqEntity,blkerr,harqEntity.TransportBlockSize,pdschInfo.G);  % retransmission, tbsize should match 
           %statusReport = updateAndAdvance(harqEntity,blkerr,harq_state(harqEntity.HARQProcessID+1,3),pdschInfo.G);
        end

        disp("Slot "+(nSlot)+". "+statusReport+ " CQI "+(mcs_k)+" SNR "+(snr_avg)+" dB"+" Doppler shift: "+(channel.MaximumDopplerShift));
        
        log_ = [log_; nSlot];

        if ~ mod(length(log_), 200)
            throughput = (1e-3)*(th/200)*((200- err)/200);
            thr = [thr, throughput];
            snr = snr/200;
            throughput_snr = 20*log(1+10^(snr/10))*5/52*((200- err)/200);
            thr_snr = [thr_snr, throughput_snr];
            th = 0;
            err = 0;
            snr = 0;
        end

        if contains(statusReport,'Initial')
            recv(nSlot+1) = {decbits};
            nSlot = nSlot + 1;    % Next slot
            vid_track = vid_track + trBlkSizes;  % track the video transmission position
        end

         %% Retransmission
         if contains(statusReport,'Retransmission')
             if contains(statusReport,'RV=2')
                 recv(log_(length(log_)-NHARQProcesses)+1) = {decbits};
             end
             if contains(statusReport,'RV=3')
                 recv(log_(length(log_)-NHARQProcesses*2)+1) = {decbits};
             end
             if contains(statusReport,'RV=1')
                 recv(log_(length(log_)-NHARQProcesses*3)+1) = {decbits};
             end
         end
         if vid_track >= vid_len
            cnt_down = cnt_down - 1;
         end
         
         if cnt_down == 0
             break;
         end
         if end_flag
             break;
         end

bler_all = length(bler(bler == 1))/length(bler);  
delay = ceil(length(log_)/pay_load);

end

rx = [];
 for ii = 1:length(recv)
     rx = [rx;cell2mat(recv(ii))];
 end
 
[~,ratio] = biterr(rx(1:vid_len),tx(1:vid_len));

end  