%% Clear workspace
clear; clc;
%% Step 1: Location and Filter for Dataset
DATA_DIR_PATH = '..\H3';
%Find all .mat files starting with Tagged_* or Testing_*

DATA_FILE_FILTER = 'Tagged\w*.mat';  % Training Files
% DATA_FILE_FILTER = 'Testing\w*.mat';  % Testing Files

%% Step 2: Get all file names under the specified folder & subfolders with regex filter	
fileList = getAllFiles(DATA_DIR_PATH, DATA_FILE_FILTER);
fprintf(1,'Found %d files matching %s at %s\n', size(fileList,2), DATA_FILE_FILTER, DATA_DIR_PATH);
for i = 1:size(fileList,2)
    [~,fname,~] = fileparts(fileList{i});
    fprintf(1,'%d. %s\n', i, fname);
end
%% Step 3: Load Data File
% Load one of training files, in partuclar the first.
fname = fileList{1}; % ***** Note: We are only loading one of the files. This should be put in a loop as needed **
clear Buffer;

fprintf(1, 'Loading file: %s\n', fname);
load(fname);
fprintf(1, 'Done loading file.\n');
%%
% For each of 2 phases in a home, compute real, reactive and apparent power from 60Hz and it's harmonic
% content. Also computes power factor. Total power in a home is sum of real
% power on both phases.
ProcessedData = Belkin_ProcessRawData(Buffer);

% Clear Buffer as we will not be using harmonic content for now.
% If you see fit, additional features, such as harmonics etc. can be
% computed from the raw Buffer.
clear Buffer;

%% Step 4: Plot data
% Plot all available data in file. The second argument controls if the
% labels are plotted.
Belkin_PlotData(ProcessedData, true);

%% Step 4 (alternative): Plot Data between first ON tagging event and last OFF tagging event
min_ts = min(cellfun(@(x)x(1), ProcessedData.TaggingInfo(:,3)));
max_ts = max(cellfun(@(x)x(1), ProcessedData.TaggingInfo(:,4)));

% Plot data between a start and stop time stamp
Belkin_PlotData(ProcessedData, true, min_ts, max_ts);

