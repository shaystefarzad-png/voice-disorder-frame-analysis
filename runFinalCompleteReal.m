function runFinalCompleteReal
% Real-data invocation of the already MATLAB-validated, frozen source.
root=fileparts(mfilename('fullpath'));collect=strcmp(getenv('VOICE_COMPLETE_COLLECT'),'1');
diseases=["Rekurrensparese","Hyperfunktionelle","Laryngitis", ...
    ("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"];
expected="9500988d523447c198ffd4ee69e983b0404888b97a06424375ea1853eac5e184";
assert(string(realSHA(fullfile(root,'VoiceDiseaseModeling_Final_Complete.m')))==expected,'Validated model source changed.');
input=fullfile(root,'datasets','approved-cycle43-v5');manifest=jsondecode(fileread(fullfile(input,'manifest.json')));
for j=1:numel(manifest),assert(strcmp(realSHA(fullfile(input,manifest(j).File)),manifest(j).PackedSHA256),'Input integrity mismatch.');end
if collect
    output=fullfile(root,'final-complete-results');active="";index=0;
    % Source/data/settings signatures match; collect only already-complete cases.
    assert(numel(dir(fullfile(output,'Repeat_001','*','COMPLETE.mat')))==6,'Six completed cases required.');
else
    index=str2double(getenv('VOICE_DISEASE_INDEX'));assert(ismember(index,1:6));active=diseases(index);
    output=fullfile(root,'final-complete-real',sprintf('disease_%d',index));
end
if ~isfolder(output),mkdir(output);end
options=struct('InputDir',input,'OutputDir',output,'Diseases',diseases,'ActiveDisease',active, ...
    'RepeatIndices',1,'PilotMode',true,'NumRepeats',100,'UseParallel',true,'Resume',true,'QuickTest',false, ...
    'Stage',"all",'WarmStartRoot',fullfile(root,'final-complete-warm'),'RunBudgetHours',5);
attempt=str2double(getenv('VOICE_COMPLETE_ATTEMPT'));if isnan(attempt),attempt=1;end
invocation=struct('DiseaseIndex',index,'Attempt',attempt,'SourceSHA',expected,'Repeat',1,'QuickTest',false);
writeJSON(fullfile(output,'Invocation.json'),invocation);
diary(fullfile(output,'Real_Execution_Log.txt'));guard=onCleanup(@()diary('off')); %#ok<NASGU>
started=tic;VoiceDiseaseModeling_Final_Complete(options);
status=readtable(fullfile(output,'Run_Status.xlsx'),'TextType','string');
complete=height(status)==choose(collect,6,1)&&all(status.Status=="Completed_all");
paused=any(status.Status=="PausedBudget");
outcome=struct('DiseaseIndex',index,'Attempt',attempt,'SourceSHA',expected,'ElapsedSeconds',toc(started), ...
    'Complete',complete,'PausedBudget',paused,'States',status.Status,'Details',status.Detail,'RepeatedSplitsExecuted',1);
writeJSON(fullfile(output,'CaseOutcome.json'),outcome);
assert(complete||paused,'Real modeling failed; inspect LastFailure.mat and execution log.');
if complete,fprintf('FINAL_COMPLETE_REAL_DATA_COMPLETED: disease %d\n',index);
else,fprintf('FINAL_COMPLETE_PAUSED_WITH_CHECKPOINTS: disease %d\n',index);end
end

function value=choose(condition,a,b)
if condition,value=a;else,value=b;end
end

function writeJSON(file,value)
fid=fopen(file,'w');assert(fid>=0);guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(value)); %#ok<NASGU>
end

function value=realSHA(file)
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(file,'rb');assert(fid>=0);guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true,bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end;md.update(typecast(bytes,'int8'));end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
