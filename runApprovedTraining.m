function runApprovedTraining
% Full seven-classifier/four-ranker training. Never evaluate the held-out Test.
root=fileparts(mfilename('fullpath'));
diseases=["Rekurrensparese","Hyperfunktionelle","Laryngitis", ...
    ("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"];
index=str2double(getenv('VOICE_DISEASE_INDEX'));
assert(isscalar(index)&&isfinite(index)&&index==fix(index)&&index>=1&&index<=numel(diseases),'Invalid disease index.');
input=fullfile(root,'datasets','approved-cycle43-v5');
output=fullfile(root,'approved-training-results',sprintf('disease_%d',index));
if ~isfolder(output),mkdir(output);end
diary(fullfile(output,'Training_Execution_Log.txt'));cleanup=onCleanup(@()diary('off')); %#ok<NASGU>
manifest=jsondecode(fileread(fullfile(input,'manifest.json')));
for j=1:numel(manifest)
    assert(strcmp(trainingSHA(fullfile(input,manifest(j).File)),manifest(j).PackedSHA256),'Input integrity check failed.');
end
options=struct('InputDir',input,'OutputDir',output,'Diseases',diseases,'ActiveDisease',diseases(index), ...
    'Stage',"training",'UseParallel',true,'Resume',true,'QuickTest',false, ...
    'MaxTrainFramesPerSubject',Inf,'MaxFolds',10,'IncludeAdaptive',true,'IncludeEqualCount',true);
VoiceDiseaseModeling_Final(options);
file=fullfile(output,'Training_Status.xlsx');
completed=false;
if isfile(file)
    status=readtable(file,'TextType','string');
    completed=height(status)==1&&all(status.Status=="TrainingOnlyCompleted");
end
if ~completed
    failures=dir(fullfile(output,'**','LastFailure.mat'));
    for j=1:numel(failures)
        saved=load(fullfile(failures(j).folder,failures(j).name),'failure');
        report=saved.failure.Report;fprintf(2,'%s\n',report);
        escaped=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
        fprintf(2,'::error title=Approved full training::%s\n',escaped);
    end
    error('Full training did not finish. Retain fold/candidate checkpoints for diagnosis or resume.');
end
assert(isempty(dir(fullfile(output,'**','FinalModel_*.mat'))),'Training stage unexpectedly created final test models.');
assert(~isempty(dir(fullfile(output,'**','TRAINING_SELECTION_FROZEN.mat'))),'Training selections were not frozen.');
summary=getenv('GITHUB_STEP_SUMMARY');
if ~isempty(summary)
    fid=fopen(summary,'a');
    if fid>=0
        fprintf(fid,'### %s\nFull training/selection completed. Seven classifiers, four rankers, every feature count.\nTest predictions: **not performed**.\n',diseases(index));
        fclose(fid);
    end
end
fprintf('FULL_TRAINING_SELECTION_COMPLETED_WITHOUT_TEST_EVALUATION: %s\n',diseases(index));
end

function value=trainingSHA(file)
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(file,'rb');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true
    bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end
    md.update(typecast(bytes,'int8'));
end
value=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
