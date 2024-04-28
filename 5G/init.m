function [carrier, pdsch, channel] = init()
do_shift = 50; % Doppler shift for the physical channel

rng(87912);  
carrier = nrCarrierConfig;
carrier.NSizeGrid = 52;

%% PDSCH and DL-SCH configuration
pdsch = nrPDSCHConfig;
pdsch.NumLayers = 2;   % Set according to DCI RI
%pdsch.PRBSet = 0:(carrier.NSizeGrid-1);     % Full band allocation
pdsch.PRBSet = 0:51;


% Set DM-RS to improve channel estimation
pdsch.DMRS.DMRSAdditionalPosition = 1;
pdsch.DMRS.DMRSConfigurationType = 1;
pdsch.DMRS.DMRSLength = 2;

%% Channel Modeling
nTxAnts = 2;   % Number of transmit antennas
nRxAnts = 2;   % Number of receive antennas

% Check that the number of layers is valid for the number of antennas
if pdsch.NumLayers > min(nTxAnts,nRxAnts)
    error("The number of layers ("+string(pdsch.NumLayers)+") must be smaller than min(nTxAnts,nRxAnts) ("+string(min(nTxAnts,nRxAnts))+")")
end

% Create a channel object
channel = nrTDLChannel;
channel.DelayProfile = "TDL-C";    % Tapped delay line channel, C type
channel.MaximumDopplerShift = do_shift;
channel.NumTransmitAntennas = nTxAnts;
channel.NumReceiveAntennas = nRxAnts;

% Set the channel sample rate to that of the OFDM signal
ofdmInfo = nrOFDMInfo(carrier);
channel.SampleRate = ofdmInfo.SampleRate;

% Channel constellation diagram
constPlot = comm.ConstellationDiagram;                                          % Constellation diagram object
constPlot.ReferenceConstellation = getConstellationRefPoints(pdsch.Modulation); % Reference constellation values
constPlot.EnableMeasurements = 1;       
% Enable EVM measurements

end