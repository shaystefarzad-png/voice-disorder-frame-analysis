function runAdaptiveFSValidation
root=fileparts(mfilename('fullpath'));artifacts=fullfile(root,'adaptive-fs-validation');
if ~isfolder(artifacts),mkdir(artifacts);end
VoiceDiseaseModeling_AdaptiveFS(struct('Stage',"selftest",'UseParallel',false));
input=fullfile(artifacts,'SyntheticFeatures');if ~isfolder(input),mkdir(input);end
rng(1405,'twister');makeSyntheticExcel(input);
options=struct('InputDir',input,'OutputDir',fullfile(artifacts,'Output'), ...
    'Diseases',"Hyperfunktionelle",'UseParallel',true,'QuickTest',true,'Resume',true, ...
    'PilotMode',true,'IncludeComparableCount',true,'RunBudgetHours',1);
VoiceDiseaseModeling_AdaptiveFS(options);
qdir=fullfile(options.OutputDir,'QUICK_TEST_NOT_FOR_INFERENCE');
status=readtable(fullfile(qdir,'Run_Status.xlsx'),'TextType','string');
assert(height(status)==1&&status.Status=="Completed_all",'Integration failed; inspect LastFailure.mat.');
files=dir(fullfile(qdir,'**','COMPLETE.mat'));assert(numel(files)==1);
saved=load(fullfile(files.folder,files.name),'result');r=saved.result;
assert(height(r.Summary)==4&&all(r.Summary.Design=="D3_Adaptive"));
assert(all(isfinite(r.Summary.TestAccuracy))&&all(isfinite(r.Summary.TestBalancedAccuracy)));
assert(numel(unique(r.WindowProbeScores.Classifier))==7&&numel(unique(r.WindowProbeScores.Feature))==43);
assert(numel(r.Searches)==5);
for j=1:5
 sc=r.Searches{j};assert(numel(unique(sc.Candidates.Classifier))==7);
 assert(numel(unique(sc.Candidates.Ranker))==4&&all(sc.Candidates.K>=1));
 assert(any(sc.Candidates.Valid)&&~any(sc.Candidates.Ranker=="None"));
 if j>=3
  t=sc.Candidates;assert(all(t.Valid==(t.CVValid&t.RefitFeasible)));
  assert(sc.Best.K<=sc.Best.RefitCapacity&&sc.Best.RefitFeasible);
 end
end
cohort=r.Cohort;assert(~any(cohort.Test&cohort.Train)&&all(cohort.Fold(cohort.Test)==0));
models=dir(fullfile(files.folder,'Model_*.mat'));
for j=1:numel(models)
 a=load(fullfile(models(j).folder,models(j).name),'bundle');b=a.bundle;
 assert(isempty(intersect(b.UsedTrainingSubjects,cohort.SubjectID(cohort.Test))));
 assert(numel(b.Columns)>=1&&numel(b.Columns)<=numel(b.WindowFeatures));
end
assert(all(ismember(["Accuracy","BalancedAccuracy"],r.PairedStatistics.Metric))&&height(r.PairedStatistics)==8);
assert(~isempty(dir(fullfile(files.folder,'Figures','*.png'))));
for j=1:height(r.Summary)
 row=r.Summary(j,:);p=r.Predictions;h=p.(row.Design+"_"+row.Scenario+"_Prediction");
 assert(abs(mean(h==p.Label)-row.TestAccuracy)<1e-12);
 assert(abs(.5*(mean(h(p.Label==1))+mean(~h(p.Label==0)))-row.TestBalancedAccuracy)<1e-12);
end
p=r.Predictions;t=p.D3_Adaptive_Time25_Branch_Prediction;f=p.D3_Adaptive_Frequency75_Branch_Prediction;
h=p.D3_Adaptive_S3B_Voting_Prediction;assert(all(h(t==f)==t(t==f)));
matfiles=dir(fullfile(qdir,'**','*.mat'));before=checkpointState(matfiles);
VoiceDiseaseModeling_AdaptiveFS(options);after=checkpointState(matfiles);assert(isequal(before,after),'Resume mutated completed models.');
writeText(fullfile(artifacts,'PASS.txt'),'Adaptive window selection then four rankers and cumulative feature counts; five searches, seven models, grouped splits, metrics, voting agreement and resume passed. Synthetic only.');
fprintf('ADAPTIVE_FS_SYNTHETIC_INTEGRATION_PASSED\n');
end

function makeSyntheticExcel(input)
names=["ZCR","ZCR_Var","RMS","PeakAmp","Entropy","EnergyEntropy", ...
    "JitterLocal_CycleOrProxy","JitterRAP_CycleOrProxy","ShimmerLocal_CycleOrProxy","ShimmerAPQ5_CycleOrProxy", ...
    "HNR","SpecCentroid","SpecRolloff","SpecFlux","SpecEntropy", ...
    "MFCC"+string(1:13),"DeltaMFCC"+string(1:13),"FormantF1","FormantF2"];
for label=0:1
    if label==0, disease="Healthy"; else, disease="Hyperfunktionelle"; end
    short=table(); long=table();
    for person=1:30
        id=string(1000+person+1000*label); subject=randn(1,43);
        for vowel=["a","i","u"]
            filename=disease+"_"+id+"-"+vowel+"_n";
            for ms=[25,75]
                n=12; offset=.01246875;
                if ms==75, n=7; offset=.03746875; end
                start=(0:n-1)'*.01;
                % Neither window nor the hybrid is designed to win.
                x=.5*subject+.8*randn(n,43)+.4*label;
                x(rand(size(x))<.015)=NaN;
                t=table(repmat(filename,n,1),repmat(id,n,1),repmat(vowel,n,1), ...
                    repmat(label,n,1),repmat("synthetic",n,1),repmat(disease,n,1), ...
                    (1:n)',start,start+offset,true(n,1),true(n,1),ones(n,1), ...
                    'VariableNames',{'FileName','SubjectID','Vowel','Label','Gender','Disease', ...
                    'FrameID','FrameStartTime','FrameCenterTime','IsActive','IsVoiced','VoicingConfidence'});
                t=[t,array2table(x,'VariableNames',cellstr(names))];
                if ms==25, short=[short;t]; else, long=[long;t]; end %#ok<AGROW>
            end
        end
    end
    % Additional unmatched and deliberately shifted recordings must be logged
    % and excluded without preventing the valid cohort from completing.
    extra=short(1:12,:); extra.FileName(:)=disease+"_unmatched";
    short=[short;extra];
    bad25=short(1:12,:); bad25.FileName(:)=disease+"_bad_timing";
    bad75=long(1:7,:); bad75.FileName(:)=disease+"_bad_timing";
    bad75.FrameStartTime=bad75.FrameStartTime+100;
    bad75.FrameCenterTime=bad75.FrameCenterTime+100;
    short=[short;bad25]; long=[long;bad75];
    writetable(short,fullfile(input,"FullFeats_25ms_"+disease+".xlsx"));
    writetable(long,fullfile(input,"FullFeats_75ms_"+disease+".xlsx"));
end
end

function x=checkpointState(files)
x=zeros(numel(files),2);
for j=1:numel(files)
    file=fullfile(files(j).folder,files(j).name); now=dir(file);
    x(j,:)=[now.bytes,now.datenum];
end
end

function s=escapePath(path)
s=replace(string(path),"'","''");
end

function writeText(file,text)
fid=fopen(file,'w'); assert(fid~=-1); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,char(text),'char');
end
