function runRealSVMPilot
root=fileparts(mfilename('fullpath'));s=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
s=strrep(s,'function VoiceDiseaseModeling_Final','function RealSVMPilotHarness');
s=strrep(s,'cfg = configuration();','realPilot(); return; cfg = configuration();');
s=[s newline fileread(fullfile(root,'ci','realSVMPilot.m.fragment'))];
w=tempname;mkdir(w);f=fopen(fullfile(w,'RealSVMPilotHarness.m'),'w');fwrite(f,s,'char');fclose(f);
addpath(w,'-begin');cleanup=onCleanup(@()rmpath(w)); %#ok<NASGU>
try,RealSVMPilotHarness;
catch err
 report=getReport(err,'extended','hyperlinks','off');
 report=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
 fprintf(2,'::error title=Real SVM pilot failure::%s\n',report);rethrow(err);
end
end
