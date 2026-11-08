function runApprovedBenchmark
% Timing on the NEW audited feature extraction. NEVER evaluate Test here.
root=fileparts(mfilename('fullpath'));input=fullfile(root,'datasets','approved-cycle43-v5');
assert(isfolder(input),'Audited inputs have not been uploaded.');
output=fullfile(root,'approved-training-benchmark');if ~isfolder(output),mkdir(output);end
manifest=jsondecode(fileread(fullfile(input,'manifest.json')));
for i=1:numel(manifest)
    file=fullfile(input,manifest(i).File);
    assert(strcmp(benchmarkSHA(file),manifest(i).PackedSHA256),'Packed feature hash mismatch.');
end
for cap=[30,Inf]
    tag="all_frames";if isfinite(cap),tag="pilot_30_frames_per_subject";end
    destination=fullfile(output,tag);
    options=struct('InputDir',input,'OutputDir',destination,'Diseases',"Hyperfunktionelle", ...
        'Stage',"benchmark",'MaxTrainFramesPerSubject',cap,'UseParallel',true,'Resume',true);
    log=evalc('VoiceDiseaseModeling_Final(options)');
    fid=fopen(fullfile(output,tag+"_execution.txt"),'w');assert(fid>=0);fprintf(fid,'%s',log);fclose(fid);
    statusFile=fullfile(destination,'Training_Status.xlsx');
    if isfile(statusFile)
        status=readtable(statusFile,'TextType','string');
        assert(all(status.Status=="TrainingOnlyCompleted"),'Benchmark did not complete.');
    else
        failures=dir(fullfile(destination,'**','LastFailure.mat'));
        for j=1:numel(failures)
            saved=load(fullfile(failures(j).folder,failures(j).name),'failure');
            report=saved.failure.Report;fprintf(2,'%s\n',report);
            report=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
            fprintf(2,'::error title=Approved training benchmark::%s\n',report);
        end
        error('Training-only benchmark failed. Inspect retained outputs.');
    end
end
fprintf('APPROVED_DATA_TRAINING_ONLY_BENCHMARK_COMPLETED_NO_TEST_PREDICTIONS\n');
end

function hash=benchmarkSHA(path)
md=java.security.MessageDigest.getInstance('SHA-256');fid=fopen(path,'rb');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true
    bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end
    md.update(typecast(bytes,'int8'));
end
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
