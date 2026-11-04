function runFinalStage
% Distributed execution of production local functions; no alternative model logic.
root=fileparts(mfilename('fullpath'));source=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
source=strrep(source,sprintf('\r\n'),sprintf('\n'));header=sprintf('function VoiceDiseaseModeling_Final\n');
assert(startsWith(source,header));
source=[header sprintf('ciStage(); return;\n') source(numel(header)+1:end) newline ...
    fileread(fullfile(root,'ci','finalStages.m.fragment'))];
folder=fullfile(root,'.ci-runtime');if ~isfolder(folder),mkdir(folder);end
file=fullfile(folder,'VoiceDiseaseModeling_Final.m');fid=fopen(file,'w');assert(fid~=-1);fwrite(fid,source,'char');fclose(fid);
old=pwd;cleanup=onCleanup(@()cd(old)); %#ok<NASGU>
cd(folder);addpath(folder,'-begin');clear VoiceDiseaseModeling_Final;
try
    VoiceDiseaseModeling_Final;
catch err
    report=string(getReport(err,'extended','hyperlinks','off'));
    report=replace(replace(replace(report,"%","%25"),char(13),"%0D"),char(10),"%0A");
    fprintf(2,'::error title=Final research stage failure::%s\n',report);
    rethrow(err);
end
end
