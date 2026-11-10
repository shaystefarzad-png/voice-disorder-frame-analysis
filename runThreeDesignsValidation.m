function runThreeDesignsValidation
% Synthetic validation only; cannot establish research effectiveness.
root=fileparts(mfilename('fullpath'));artifacts=fullfile(root,'three-designs-validation');
if ~isfolder(artifacts),mkdir(artifacts);end
VoiceDiseaseModeling_ThreeDesigns(struct('Stage',"selftest",'UseParallel',false));
work=tempname;mkdir(work);input=fullfile(work,'Features');mkdir(input);
rng(1405,'twister');makeSyntheticExcel(input);
options=struct('InputDir',input,'OutputDir',fullfile(artifacts,'Output'),'Diseases',"Hyperfunktionelle", ...
    'UseParallel',true,'QuickTest',true,'Resume',true,'PilotMode',true,'NumRepeats',100);
VoiceDiseaseModeling_ThreeDesigns(options);
qdir=fullfile(options.OutputDir,'QUICK_TEST_NOT_FOR_INFERENCE');
status=readtable(fullfile(qdir,'Run_Status.xlsx'),'TextType','string');
assert(height(status)==1&&status.Status=="Completed_all",'Synthetic workflow failed; see LastFailure.mat and annotations.');
files=dir(fullfile(qdir,'**','COMPLETE.mat'));assert(numel(files)==1);
saved=load(fullfile(files.folder,files.name),'result');r=saved.result;
assert(height(r.Summary)==12&&numel(unique(r.Summary.Design))==3);
assert(all(isfinite(r.Summary.TestAccuracy)));
assert(all(r.Summary.FeatureCount(r.Summary.Design=="D2_NoFS"&r.Summary.Scenario~="S3B_Voting")==43));
assert(all(r.Summary.FeatureCount(r.Summary.Design=="D3_Adaptive"&r.Summary.Scenario~="S3B_Voting")==43));
assert(height(r.EqualCountSummary)==4&&numel(unique(r.EqualCountSummary.FeatureCount))==1);
cohort=r.Cohort;assert(height(cohort)==60&&sum(cohort.Test)==12&&sum(cohort.Train)==48);
assert(~any(cohort.Test&cohort.Train)&&all(cohort.Fold(cohort.Test)==0));
p=r.Predictions;assert(height(p)==12&&isequal(sort(p.SubjectID),sort(cohort.SubjectID(cohort.Test))));
for j=1:height(r.Summary)
    row=r.Summary(j,:);h=p.(row.Design+"_"+row.Scenario+"_Prediction");
    assert(abs(mean(h==p.Label)-row.TestAccuracy)<1e-12);
    ba=.5*(mean(h(p.Label==1))+mean(~h(p.Label==0)));assert(abs(ba-row.TestBalancedAccuracy)<1e-12);
end
for j=1:numel(r.Searches)
    sc=r.Searches{j};assert(numel(unique(sc.Candidates.Classifier))==7);
    if startsWith(r.SearchNames(j),"D1_FS"),assert(numel(unique(sc.Candidates.Ranker))==4&&all(sc.Candidates.Valid));
    else,assert(all(sc.Candidates.Ranker=="None")&&all(sc.Candidates.K==0));end
end
for f=["D1_FS","D2_NoFS","D3_Adaptive"]
    vote=r.Voting.(f);assert(height(vote.OuterChoices)==2&&vote.FinalChoice.SelectionUsesTrainingOnly);
    if f~="D1_FS",assert(all(vote.OuterChoices.TimeFeatures+vote.OuterChoices.FrequencyFeatures==43));end
end
models=dir(fullfile(files.folder,'Model_*.mat'));
for j=1:numel(models)
    b=load(fullfile(models(j).folder,models(j).name),'bundle');b=b.bundle;
    assert(all(ismember(b.UsedTrainingSubjects,cohort.SubjectID(cohort.Train))));
    assert(isempty(intersect(b.UsedTrainingSubjects,cohort.SubjectID(cohort.Test))));
end
stats=readtable(fullfile(qdir,'Statistical_Hypothesis_Validation.xlsx'),'Sheet','PerDisease');
assert(height(stats)==12&&all(isnan(stats.P_TwoSided))&&~any(stats.IndependentConfirmation));
assert(isfile(fullfile(files.folder,'ALL_TRAINING_CHOICES_FROZEN.mat')));
matfiles=dir(fullfile(qdir,'**','*.mat'));before=checkpointState(matfiles);
VoiceDiseaseModeling_ThreeDesigns(options);after=checkpointState(matfiles);assert(isequal(before,after),'Resume changed completed models.');
writeText(fullfile(artifacts,'PASS.txt'),'All three designs, seven models, four rankers, equal feature budgets, nested adaptive voting and resume passed. Synthetic only.');
fprintf('THREE_DESIGNS_SYNTHETIC_INTEGRATION_PASSED\n');
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
