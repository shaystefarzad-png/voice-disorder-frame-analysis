function runHyperTrainingDevelopment
% Separate exploratory experiment. Never evaluates the original 44 Test subjects.
root=fileparts(mfilename('fullpath'));setenv('HYPER_DEV_ROOT',root);
source=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
source=strrep(source,'function VoiceDiseaseModeling_Final','function HyperDevelopmentHarness');
source=strrep(source,'cfg = configuration();','hdStage(); return; cfg = configuration();');
source=[source newline fileread(fullfile(root,'ci','hyperDevelopment.m.fragment'))];
work=fullfile(root,'.hyper-dev-runtime');if ~isfolder(work),mkdir(work);end
fid=fopen(fullfile(work,'HyperDevelopmentHarness.m'),'w');assert(fid>=0);fwrite(fid,source,'char');fclose(fid);
addpath(work,'-begin');cleanup=onCleanup(@()rmpath(work)); %#ok<NASGU>
try
    HyperDevelopmentHarness;
catch err
    report=getReport(err,'extended','hyperlinks','off');
    report=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
    fprintf(2,'::error title=Training-only hyperfunctional development::%s\n',report);rethrow(err);
end
end
