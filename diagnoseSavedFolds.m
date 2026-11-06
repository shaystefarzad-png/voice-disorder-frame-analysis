function diagnoseSavedFolds
files=dir(fullfile('recovery','**','Fold_*.mat'));
assert(~isempty(files),'No checkpoint evidence was restored.');
out=fullfile('recovery','reports');if ~isfolder(out),mkdir(out);end
summary=table();failures=table();
for j=1:numel(files)
    z=load(fullfile(files(j).folder,files(j).name),'part');p=z.part;
    q=table(string(files(j).folder),string(files(j).name),numel(p.BA),sum(isfinite(p.BA)), ...
        height(p.Failures),'VariableNames',{'CheckpointFolder','File','CandidateCount','FiniteResults','Failures'});
    summary=[summary;q]; %#ok<AGROW>
    f=p.Failures;
    f.Checkpoint=repmat(string(files(j).folder)+"/"+string(files(j).name),height(f),1);
    failures=[failures;f]; %#ok<AGROW>
end
writetable(summary,fullfile(out,'Checkpoint_Status.csv'));
writetable(failures,fullfile(out,'Failed_Candidates.csv'));
fprintf('PRESERVED_CHECKPOINT_DIAGNOSIS_COMPLETE; NO MODELS REFIT, NO TEST EVALUATION\n');
disp(summary);
end
