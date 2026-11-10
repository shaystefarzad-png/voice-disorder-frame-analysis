function runThreeDesignsPilot
root=fileparts(mfilename('fullpath'));
diseases=["Rekurrensparese","Hyperfunktionelle","Laryngitis", ...
    ("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"];
index=str2double(getenv('VOICE_DISEASE_INDEX'));assert(ismember(index,1:6));
output=fullfile(root,'three-designs-pilot-results',sprintf('disease_%d',index));if ~isfolder(output),mkdir(output);end
options=struct('InputDir',fullfile(root,'datasets','approved-cycle43-v5'),'OutputDir',output, ...
    'Diseases',diseases,'ActiveDisease',diseases(index),'RepeatIndices',1,'PilotMode',true,'NumRepeats',100, ...
    'UseParallel',true,'Resume',true,'QuickTest',false,'Stage',"all", ...
    'WarmStartRoot',fullfile(root,'three-designs-warm'),'RunBudgetHours',5);
timer=tic;VoiceDiseaseModeling_ThreeDesigns(options);elapsed=toc(timer);
fid=fopen(fullfile(output,'PilotTiming.json'),'w');assert(fid>=0);
fprintf(fid,'%s',jsonencode(struct('Disease',diseases(index),'ElapsedSeconds',elapsed,'ExecutedRepeat',1,'PlannedRepeats',100, ...
    'WarmStartOfCompatibleTrainingScores',true,'HundredRepeatRuntimeNotMeasured',true)));fclose(fid);
s=readtable(fullfile(output,'Run_Status.xlsx'),'TextType','string');
assert(height(s)==1&&all(s.Status=="Completed_all"),'Pilot incomplete; retained all candidate checkpoints and failure report.');
fprintf('THREE_DESIGNS_REAL_PILOT_COMPLETED: %s | %.1f seconds\n',diseases(index),elapsed);
end
