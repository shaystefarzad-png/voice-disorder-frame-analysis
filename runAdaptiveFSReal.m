function runAdaptiveFSReal
% New experiment; never restores any preceding D1/D2/D3 checkpoint.
root=fileparts(mfilename('fullpath'));
diseases=["Rekurrensparese","Hyperfunktionelle","Laryngitis", ...
    ("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"];
index=str2double(getenv('VOICE_DISEASE_INDEX'));assert(ismember(index,1:6));
input=fullfile(root,'datasets','approved-cycle43-v5');
manifest=jsondecode(fileread(fullfile(input,'manifest.json')));
for j=1:numel(manifest)
    assert(strcmp(localSHA(fullfile(input,manifest(j).File)),manifest(j).PackedSHA256),'Input integrity mismatch.');
end
output=fullfile(root,'adaptive-fs-real-new',sprintf('disease_%d',index));
if ~isfolder(output),mkdir(output);end
options=struct('InputDir',input,'OutputDir',output,'Diseases',diseases,'ActiveDisease',diseases(index), ...
    'RepeatIndices',1,'PilotMode',true,'UseParallel',true,'Resume',true,'QuickTest',false, ...
    'Stage',"all",'WarmStartRoot',"",'RunBudgetHours',5,'AdaptiveFeatureSelection',true, ...
    'IncludeComparableCount',true,'ActiveDesigns',3,'ResumeSourceSHA',"",'ResumeExecutionSHA',"");
invocation=struct('Experiment',"Adaptive_Window_Then_CumulativeFS_v1",'DiseaseIndex',index, ...
    'SourceSHA',localSHA(fullfile(root,'VoiceDiseaseModeling_AdaptiveFS.m')), ...
    'PreviousExperimentCheckpointsAllowed',false,'SelectionMetric',"CV Accuracy", ...
    'Repeat',1,'QuickTest',false,'AdaptiveFeatureSelection',true);
writeJSON(fullfile(output,'Invocation.json'),invocation);
diary(fullfile(output,'Real_Execution_Log.txt'));guard=onCleanup(@()diary('off')); %#ok<NASGU>
timer=tic;VoiceDiseaseModeling_AdaptiveFS(options);
status=readtable(fullfile(output,'Run_Status.xlsx'),'TextType','string');
complete=height(status)==1&&all(status.Status=="Completed_all");paused=any(status.Status=="PausedBudget");
writeJSON(fullfile(output,'CaseOutcome.json'),struct('DiseaseIndex',index, ...
    'ElapsedSeconds',toc(timer),'Complete',complete,'PausedBudget',paused, ...
    'States',status.Status,'Details',status.Detail,'SourceSHA',invocation.SourceSHA));
assert(complete||paused,'New experiment failed; inspect LastFailure.mat.');
if complete
    saved=dir(fullfile(output,'Repeat_001','*','COMPLETE.mat'));assert(numel(saved)==1);
    s=load(fullfile(saved.folder,saved.name),'result');r=s.result;
    assert(height(r.Summary)==4&&all(r.Summary.Design=="D3_Adaptive"));
    for j=1:numel(r.Searches)
        t=r.Searches{j}.Candidates;assert(numel(unique(t.Classifier))==7&&numel(unique(t.Ranker))==4);
        assert(all(t.K>=1),'Adaptive experiment must have cumulative selection after window choice.');
    end
    fprintf('NEW_ADAPTIVE_FS_REAL_COMPLETED: disease %d\n',index);
else,fprintf('NEW_ADAPTIVE_FS_PAUSED_WITH_NEW_EXPERIMENT_CHECKPOINTS: disease %d\n',index);end
end
function writeJSON(file,value)
fid=fopen(file,'w');assert(fid>=0);guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(value)); %#ok<NASGU>
end
function value=localSHA(file)
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(file,'rb');assert(fid>=0);guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true,bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end;md.update(typecast(bytes,'int8'));end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
