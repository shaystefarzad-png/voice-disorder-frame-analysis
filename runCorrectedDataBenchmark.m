function runCorrectedDataBenchmark
% User-authorized real-data timing pilot; never evaluates Test.
root=fileparts(mfilename('fullpath'));output=getenv('VOICE_RESULTS_DIR');
assert(~isempty(output) && ~isempty(getenv('VOICE_FEATURE_DIR')));
if ~isfolder(output),mkdir(output);end
try
    source=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
    source=strrep(source,sprintf('\r\n'),sprintf('\n'));
    header=sprintf('function VoiceDiseaseModeling_Final\n');assert(startsWith(source,header));
    source=[header sprintf('benchmarkOnly(); return;\n') source(numel(header)+1:end) ...
        newline fileread(fullfile(root,'ci','benchmarkOnly.m.fragment'))];
    folder=fullfile(getenv('RUNNER_TEMP'),'benchmark-code');if ~isfolder(folder),mkdir(folder);end
    fid=fopen(fullfile(folder,'VoiceDiseaseModeling_Final.m'),'w');assert(fid~=-1);fwrite(fid,source,'char');fclose(fid);
    old=pwd;cleanup=onCleanup(@()cd(old)); %#ok<NASGU>
    cd(folder);addpath(folder,'-begin');clear VoiceDiseaseModeling_Final;
    log=evalc('VoiceDiseaseModeling_Final');
    fid=fopen(fullfile(output,'Benchmark_Log.txt'),'w');fwrite(fid,log,'char');fclose(fid);
    fprintf('REAL_DATA_TRAINING_BENCHMARK_PASSED_NO_TEST_EVALUATION\n');
catch err
    failure=struct('Identifier',err.identifier,'Report',getReport(err,'extended','hyperlinks','off'));
    save(fullfile(output,'Benchmark_Failure.mat'),'failure');
    fprintf(2,'%s\n',failure.Report);
    rethrow(err);
end
end
