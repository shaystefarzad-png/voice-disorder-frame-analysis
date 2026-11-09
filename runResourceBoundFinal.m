function runResourceBoundFinal
% Test is reached only after ALL SIX diseases have frozen training choices.
root=fileparts(mfilename('fullpath'));input=fullfile(root,'datasets','approved-cycle43-v5');
output=fullfile(root,'approved-training30-final');if ~isfolder(output),mkdir(output);end
sourceSHA=finalSHA(fullfile(root,'VoiceDiseaseModeling_Final.m'));
inputSHA=finalSHA(fullfile(input,'manifest.json'));
for d=1:6
    source=fullfile(root,'training30-completed',sprintf('approved-training30-d%d',d));
    specification=jsondecode(fileread(fullfile(source,'Experiment_Specification.json')));
    assert(specification.DiseaseIndex==d && specification.TrainingFrameCap==30 && string(specification.NumericalSourceSHA)==string(sourceSHA) && ...
        string(specification.InputManifestSHA)==string(inputSHA),'Final source/input mismatch.');
    frozenFiles=dir(fullfile(source,'**','TRAINING_SELECTION_FROZEN.mat'));
    assert(numel(frozenFiles)==1,'A disease has not uniquely frozen training selection.');
    frozen=load(fullfile(frozenFiles.folder,frozenFiles.name),'selection');
    assert(frozen.selection.FrozenBeforeTest && isfinite(frozen.selection.TimeChoice.MeanBA) && ...
        isfinite(frozen.selection.FrequencyChoice.MeanBA),'Invalid frozen selection.');
    assert(isempty(dir(fullfile(source,'**','LastFailure.mat'))),'A disease recorded a numerical failure.');
end
% Preserve the unchanged cache paths under a common OutputDir. OutputDir,
% Stage and ActiveDisease are execution-only and excluded from signatures.
for d=1:6
    source=fullfile(root,'training30-completed',sprintf('approved-training30-d%d',d));
    entries=dir(source);entries=entries([entries.isdir]&~startsWith(string({entries.name}),'.'));
    assert(numel(entries)==1,'Unexpected disease directory structure.');
    copyTree(fullfile(source,entries.name),fullfile(output,entries.name));
end
options=struct('InputDir',input,'OutputDir',output,'Stage',"all",'ActiveDisease',"", ...
    'UseParallel',true,'Resume',true,'QuickTest',false,'MaxTrainFramesPerSubject',30, ...
    'MaxFolds',10,'IncludeAdaptive',true,'IncludeEqualCount',true);
VoiceDiseaseModeling_Final(options);
book=fullfile(output,'All_Diseases_Summary.xlsx');
status=readtable(book,'Sheet','RunStatus','TextType','string');
assert(height(status)==6 && all(status.Status=="Completed"),'Final evaluation incomplete.');
statistics=fullfile(output,'Statistical_Hypothesis_Validation.xlsx');
report=struct('Experiment',"Predeclared Train30; all validation/Test frames", ...
    'TrainingFrameCap',30,'AllSixSelectionsFrozenBeforeTest',true, ...
    'Summary',table2struct(readtable(book,'Sheet','Summary','TextType','string')), ...
    'EqualCountSummary',table2struct(readtable(book,'Sheet','EqualCountSummary','TextType','string')), ...
    'PerDiseaseHypothesis',table2struct(readtable(statistics,'Sheet','PerDisease','TextType','string')), ...
    'OverallHypothesis',table2struct(readtable(statistics,'Sheet','Overall','TextType','string')), ...
    'Interpretation',"Results concern the CycleOrProxy hybrid recipe; adaptive/equal-count comparisons remain supplemental.");
fid=fopen(fullfile(output,'Final_Report.json'),'w','n','UTF-8');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(report)); %#ok<NASGU>
fprintf('TRAIN30_FINAL_EVALUATION_AND_PLANNED_STATISTICS_COMPLETED\n');
end

function copyTree(source,destination)
if ~isfolder(destination),mkdir(destination);end
entries=dir(source);
for j=1:numel(entries)
    name=entries(j).name;if strcmp(name,'.')||strcmp(name,'..'),continue;end
    from=fullfile(source,name);to=fullfile(destination,name);
    if entries(j).isdir,copyTree(from,to);else,copyfile(from,to,'f');end
end
end

function value=finalSHA(file)
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(file,'rb');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true
    bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end
    md.update(typecast(bytes,'int8'));
end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
