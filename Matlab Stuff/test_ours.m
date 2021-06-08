% camera's parameters
width= 240;
height= 135;
f = 293.33334351;
load sift4.mat;
K = [f 0 width
    0 f height
    0 0 1];

% pose estimation
try
      [R1,t1]= ASPnP(points3D, points2D, K);
      disp("Pred");
      disp(R1);
      disp(t1);
 catch
    disp(['The solver - ','ASPnP - encounters internal errors!!!']);
end