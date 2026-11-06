function verifyCertifiedSVM
% Exercise the actual standalone production solver, without touching Test data.
root=fileparts(mfilename('fullpath'));source=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
source=strrep(source,'function VoiceDiseaseModeling_Final','function CertifiedSolverHarness');
source=strrep(source,'cfg = configuration();','certifiedChecks(); return; cfg = configuration();');
checks=fileread(fullfile(root,'ci','certifiedChecks.m.fragment'));
work=tempname;mkdir(work);file=fullfile(work,'CertifiedSolverHarness.m');
f=fopen(file,'w');assert(f>=0);fwrite(f,[source newline checks],'char');fclose(f);
addpath(work,'-begin');cleanup=onCleanup(@()rmpath(work)); %#ok<NASGU>
try
    CertifiedSolverHarness;
catch err
    report=getReport(err,'extended','hyperlinks','off');
    report=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
    fprintf(2,'::error title=Certified SVM verification::%s\n',report);rethrow(err);
end
end
